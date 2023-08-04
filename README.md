# SDXL Refiner fixed (stable-diffusion-webui Extension)
## Extension for integration of the SDXL refiner into Automatic1111

This extension makes the SDXL Refiner available in [Automatic1111 stable-diffusion-webui.](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

The implentation is done as [described by Stability AI](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) as *an ensemble of experts pipeline for latent diffusion: In a first step, the base model is used to generate (noisy) latents, which are then further processed with a refinement model specialized for the final denoising steps.*

This extension is heavily based on [sd-webui-refiner](https://github.com/wcde/sd-webui-refiner), so thanks goes to [wcde](https://github.com/wcde) for his great work.

Nevertheless this extension has some modfications and enhancments over sd-webui-refiner

* simplyfied ui to avoid the most common misunderstanding of the relation between refiner steps and total steps. (for more details see [Handover from Base to Refiner](#Handover from Base to Refiner))
* uses a fixed point where the diffusion is handed over from the base model to the refiner. (for more details see [Handover from Base to Refiner](#Handover from Base to Refiner))
* configuration can be saved with Automatic1111 Settings -> Defaults -> View Changes -> Apply

## Installation

In Automatic1111 go to the **Extensions** tab and there to **Install from URL**. Paste this URL 

`https://github.com/lisanet/sdxl-webui-refiner-fixed.git`

into the field **URL for extension's git repository**, leave the other fields blank and hit the **Install** button. 

Now head over to the **Installed** tab, there mark the extension checked and hit **Apply and restart UI**.

The extension is now loaded and you can access it on the txt2img and img2img tabs.

## Usage

To use the refiner you need to already have downloaded the SDXL 1.0 refiner. You can [find it on Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/tree/main).

On the **txt2img** tab scroll down, open the Refiner panel, check **Enable Refiner** and select the refiner checkpoints from the **Model** dropdown. 

That's it. Now you can generate an image as usual. The extension will take care of handing over the latent image from the base checkpoints to the refiner witout the need or hassle to define the correct handover point for yourself.

The extension is available on the img2img tab too.

## Handover from Base to Refiner

As [described here](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl), *Stable Diffusion XL base is trained on timesteps 0-999 and Stable Diffusion XL refiner is finetuned from the base model on low noise timesteps 0-199 inclusive, so we use the base model for the first 800 timesteps (high noise) and the refiner for the last 200 timesteps (low noise). Hence, high_noise_frac is set to 0.8, so that all steps 200-999 (the first 80% of denoising timesteps) are performed by the base model and steps 0-199 (the last 20% of denoising timesteps) are performed by the refiner model.*

To avoid one if the most misunderstandings in the usage of the refiner, this extension sets the handover point at 80% of the denoised image and uses the refiner for the last 20%. Since this extensions has to use steps instead of timestamps, the 80% timestemp mark is not met exactly, though it's never lower. 

This fixed handover point is the main advantage, because if the handover point will be lower than 80% it's very likely, that the refiner will introduce distortions in the final image. And there are a lot of articles and videos out there, who have misunderstood the former steps slider in the extension and the relation between this value and the total steps.

If you still prefer to play around (with the risk of distorted images) and set the hand over value on your own, please have a look at the above mentioned sd-webui-refiner extension.

## Technical notes

The extension loads only UNET from the refiner checkpoint and replaces he tbase UNET with it at the handover point for last steps of denoising.

Use Tiled VAE if you have 12GB or less VRAM.
