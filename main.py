import torch
import gc
import numpy as np
from typing import List, Optional
from diffusers import QwenImageEditPlusPipeline
from diffusers.utils import load_image
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
    calculate_shift, retrieve_timesteps, calculate_dimensions,
    CONDITION_IMAGE_SIZE, VAE_IMAGE_SIZE,
)
from diffusers.pipelines.qwenimage.pipeline_output import QwenImagePipelineOutput


def patched_call(
    self,
    image=None,
    prompt=None,
    negative_prompt=None,
    true_cfg_scale: float = 4.0,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    sigmas=None,
    guidance_scale=None,
    num_images_per_prompt: int = 1,
    generator=None,
    latents=None,
    prompt_embeds=None,
    prompt_embeds_mask=None,
    negative_prompt_embeds=None,
    negative_prompt_embeds_mask=None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    attention_kwargs=None,
    callback_on_step_end=None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
):
    """Patched __call__ method that supports batch processing of multiple images."""
    
    # Handle batch of images
    if isinstance(image, list):
        batch_size = len(image)
        image_size = image[-1].size
    else:
        batch_size = 1
        image_size = image.size
        image = [image]
    
    # Handle prompts - ensure we have one prompt per image
    if isinstance(prompt, str):
        prompt = [prompt] * batch_size
    elif prompt is None and prompt_embeds is not None:
        batch_size = prompt_embeds.shape[0]
    
    # Handle negative prompts
    if isinstance(negative_prompt, str):
        negative_prompt = [negative_prompt] * batch_size
    
    calculated_width, calculated_height = calculate_dimensions(1024 * 1024, image_size[0] / image_size[1])
    height = height or calculated_height
    width = width or calculated_width

    multiple_of = self.vae_scale_factor * 2
    width = width // multiple_of * multiple_of
    height = height // multiple_of * multiple_of

    self._guidance_scale = guidance_scale
    self._attention_kwargs = attention_kwargs
    self._current_timestep = None
    self._interrupt = False

    device = self._execution_device
    dtype = self.transformer.dtype
    
    # Preprocess all images and collect condition/vae images
    all_condition_images = []
    all_vae_images = []
    all_vae_image_sizes = []
    
    for img in image:
        image_width, image_height = img.size
        condition_width, condition_height = calculate_dimensions(
            CONDITION_IMAGE_SIZE, image_width / image_height
        )
        vae_width, vae_height = calculate_dimensions(VAE_IMAGE_SIZE, image_width / image_height)
        all_vae_image_sizes.append((vae_width, vae_height))
        all_condition_images.append(self.image_processor.resize(img, condition_height, condition_width))
        all_vae_images.append(self.image_processor.preprocess(img, vae_height, vae_width).unsqueeze(2))

    has_neg_prompt = negative_prompt is not None or (
        negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
    )
    do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
    
    # Encode each image-prompt pair separately, then batch
    all_prompt_embeds = []
    all_prompt_masks = []
    all_image_latents = []
    
    with torch.no_grad():
        for i, (cond_img, vae_img, p) in enumerate(zip(all_condition_images, all_vae_images, prompt)):
            # Encode prompt with this specific image
            embeds, mask = self.encode_prompt(
                image=[cond_img],
                prompt=p,
                prompt_embeds=None,
                prompt_embeds_mask=None,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )
            
            if mask is None:
                mask = torch.ones(embeds.shape[0], embeds.shape[1], device=embeds.device, dtype=torch.long)
            
            all_prompt_embeds.append(embeds.cpu())
            all_prompt_masks.append(mask.cpu())
            
            # Encode image through VAE
            vae_img_tensor = vae_img.to(device=device, dtype=dtype)
            image_latent = self._encode_vae_image(image=vae_img_tensor, generator=generator)
            all_image_latents.append(image_latent.cpu())
            
            torch.cuda.empty_cache()
            gc.collect()
    
    # Pad embeddings to same sequence length and stack
    max_seq_len = max(e.shape[1] for e in all_prompt_embeds)
    padded_embeds = []
    padded_masks = []
    for embeds, mask in zip(all_prompt_embeds, all_prompt_masks):
        if embeds.shape[1] < max_seq_len:
            pad_len = max_seq_len - embeds.shape[1]
            embeds = torch.cat([embeds, torch.zeros(1, pad_len, embeds.shape[2], device=embeds.device, dtype=embeds.dtype)], dim=1)
            mask = torch.cat([mask, torch.zeros(1, pad_len, device=mask.device, dtype=mask.dtype)], dim=1)
        padded_embeds.append(embeds)
        padded_masks.append(mask)
    
    prompt_embeds = torch.cat(padded_embeds, dim=0).to(device=device, dtype=dtype)
    prompt_embeds_mask = torch.cat(padded_masks, dim=0).to(device=device)
    
    # Stack image latents
    image_latents = torch.cat(all_image_latents, dim=0).to(device=device, dtype=dtype)
    
    # Handle negative prompts if needed
    if do_true_cfg:
        all_neg_embeds = []
        all_neg_masks = []
        
        for i, (cond_img, neg_p) in enumerate(zip(all_condition_images, negative_prompt)):
            neg_embeds, neg_mask = self.encode_prompt(
                image=[cond_img],
                prompt=neg_p,
                prompt_embeds=None,
                prompt_embeds_mask=None,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )
            if neg_mask is None:
                neg_mask = torch.ones(neg_embeds.shape[0], neg_embeds.shape[1], device=neg_embeds.device, dtype=torch.long)
            all_neg_embeds.append(neg_embeds.cpu())
            all_neg_masks.append(neg_mask.cpu())
            torch.cuda.empty_cache()
        
        max_neg_seq_len = max(e.shape[1] for e in all_neg_embeds)
        padded_neg_embeds = []
        padded_neg_masks = []
        for embeds, mask in zip(all_neg_embeds, all_neg_masks):
            if embeds.shape[1] < max_neg_seq_len:
                pad_len = max_neg_seq_len - embeds.shape[1]
                embeds = torch.cat([embeds, torch.zeros(1, pad_len, embeds.shape[2], device=embeds.device, dtype=embeds.dtype)], dim=1)
                mask = torch.cat([mask, torch.zeros(1, pad_len, device=mask.device, dtype=mask.dtype)], dim=1)
            padded_neg_embeds.append(embeds)
            padded_neg_masks.append(mask)
        
        negative_prompt_embeds = torch.cat(padded_neg_embeds, dim=0).to(device=device, dtype=dtype)
        negative_prompt_embeds_mask = torch.cat(padded_neg_masks, dim=0).to(device=device)

    # Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels // 4
    latent_height = 2 * (int(height) // (self.vae_scale_factor * 2))
    latent_width = 2 * (int(width) // (self.vae_scale_factor * 2))
    shape = (batch_size, 1, num_channels_latents, latent_height, latent_width)
    
    from diffusers.utils.torch_utils import randn_tensor
    latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    latents = self._pack_latents(latents, batch_size, num_channels_latents, latent_height, latent_width)
    
    # Pack image latents
    image_latent_height, image_latent_width = image_latents.shape[3:]
    image_latents = self._pack_latents(
        image_latents, batch_size, num_channels_latents, image_latent_height, image_latent_width
    )
    
    # Build img_shapes for each batch item
    img_shapes = [
        [
            (1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2),
            (1, vae_h // self.vae_scale_factor // 2, vae_w // self.vae_scale_factor // 2),
        ]
        for vae_w, vae_h in all_vae_image_sizes
    ]

    # Prepare timesteps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        self.scheduler.config.get("base_image_seq_len", 256),
        self.scheduler.config.get("max_image_seq_len", 4096),
        self.scheduler.config.get("base_shift", 0.5),
        self.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    self._num_timesteps = len(timesteps)

    # Handle guidance
    if self.transformer.config.guidance_embeds and guidance_scale is None:
        raise ValueError("guidance_scale is required for guidance-distilled model.")
    elif self.transformer.config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])
    else:
        guidance = None

    if self.attention_kwargs is None:
        self._attention_kwargs = {}

    # Denoising loop
    self.scheduler.set_begin_index(0)
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            self._current_timestep = t

            latent_model_input = latents
            if image_latents is not None:
                latent_model_input = torch.cat([latents, image_latents], dim=1)

            timestep = t.expand(latents.shape[0]).to(latents.dtype)
            with self.transformer.cache_context("cond"):
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    encoder_hidden_states=prompt_embeds,
                    img_shapes=img_shapes,
                    attention_kwargs=self.attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred[:, : latents.size(1)]

            if do_true_cfg:
                with self.transformer.cache_context("uncond"):
                    neg_noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states_mask=negative_prompt_embeds_mask,
                        encoder_hidden_states=negative_prompt_embeds,
                        img_shapes=img_shapes,
                        attention_kwargs=self.attention_kwargs,
                        return_dict=False,
                    )[0]
                neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
                comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                noise_pred = comb_pred * (cond_norm / noise_norm)

            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    latents = latents.to(latents_dtype)

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

    self._current_timestep = None
    if output_type == "latent":
        output_image = latents
    else:
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        output_image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
        output_image = self.image_processor.postprocess(output_image, output_type=output_type)

    self.maybe_free_model_hooks()

    if not return_dict:
        return (output_image,)

    return QwenImagePipelineOutput(images=output_image)


if __name__ == "__main__":
    # Monkey-patch the class's __call__ method BEFORE creating the pipeline
    QwenImageEditPlusPipeline._original_call = QwenImageEditPlusPipeline.__call__
    QwenImageEditPlusPipeline.__call__ = torch.no_grad()(patched_call)
    
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2511",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    pipe.load_lora_weights("prithivMLmods/QIE-2511-Object-Remover-v2")

    input_paths = [
        "/workspace/layered-segmentation/output/office/qwen/overlaid/book__under__telescope_08_input.png",
        "/workspace/layered-segmentation/output/office/qwen/overlaid/bottle_09_input.png",
    ]
    input_images = [load_image(path) for path in input_paths]

    prompt = "Remove the red highlighted object from the scene."
    prompts = [prompt] * len(input_images)

    # Now supports batch processing!
    images = pipe(image=input_images, prompt=prompts).images

    for i, img in enumerate(images):
        img.save(f"./qwen_image_edit_output_{i}.png")
