import hydra
import torch
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
import itertools
from torch.utils.data import DataLoader
from tqdm import tqdm
from diffusers import LMSDiscreteScheduler
from diffusers.optimization import get_scheduler
import math
from my_model import unet_2d_condition
import os
import json
from accelerate.logging import get_logger
import hashlib
from torch.utils.data import Dataset
from torchvision import transforms
from utils import setup_logger, load_text_inversion
from accelerate.utils import set_seed

class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
            self,
            instance_data_root,
            instance_prompt,
            tokenizer,
            class_data_root=None,
            class_prompt=None,
            size=512,
            center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

        return example

class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example

def train_dreambooth(device, unet, vae, tokenizer, text_encoder, cfg, accelerator, logger):
    if cfg.dreambooth.with_prior_preservation:
        class_images_dir = Path(cfg.dreambooth.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < cfg.dreambooth.num_class_images:
            torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
            pipeline = StableDiffusionPipeline.from_pretrained(
                cfg.dreambooth.pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                safety_checker=None,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = cfg.dreambooth.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(cfg.dreambooth.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=cfg.dreambooth.sample_batch_size)

            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)

            for example in tqdm(
                    sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    vae.requires_grad_(False)
    if not cfg.dreambooth.train_text_encoder:
        text_encoder.requires_grad_(False)

    if cfg.dreambooth.scale_lr:
        cfg.dreambooth.lr = (
                cfg.dreambooth.lr * cfg.dreambooth.gradient_accumulation_steps * cfg.dreambooth.train_batch_size * accelerator.num_processes
        )
    optimizer_class = torch.optim.AdamW
    params_to_optimize = (
        itertools.chain(unet.parameters(), text_encoder.parameters()) if cfg.dreambooth.train_text_encoder else unet.parameters()
    )
    optimizer = optimizer_class(
        params_to_optimize,
        lr=cfg.dreambooth.lr,
        betas=(cfg.dreambooth.adam_beta1, cfg.dreambooth.adam_beta2),
        weight_decay=cfg.dreambooth.adam_weight_decay,
        eps=cfg.dreambooth.adam_epsilon,
    )
    noise_scheduler = DDPMScheduler.from_config(cfg.dreambooth.pretrained_model_name_or_path, subfolder="scheduler")
    train_dataset = DreamBoothDataset(
        instance_data_root=cfg.dreambooth.instance_data_dir,
        instance_prompt=cfg.dreambooth.instance_prompt,
        class_data_root=cfg.dreambooth.class_data_dir if cfg.dreambooth.with_prior_preservation else None,
        class_prompt=cfg.dreambooth.class_prompt,
        tokenizer=tokenizer,
        size=cfg.dreambooth.resolution,
        center_crop=cfg.dreambooth.center_crop,
    )
    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if cfg.dreambooth.with_prior_preservation:
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad(
            {"input_ids": input_ids},
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }
        return batch

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.dreambooth.train_batch_size, shuffle=True, collate_fn=collate_fn, num_workers=1
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.dreambooth.gradient_accumulation_steps)
    if cfg.dreambooth.max_train_steps is None:
        cfg.dreambooth.max_train_steps = cfg.dreambooth.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        cfg.dreambooth.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.dreambooth.lr_warmup_steps * cfg.dreambooth.gradient_accumulation_steps,
        num_training_steps=cfg.dreambooth.max_train_steps * cfg.dreambooth.gradient_accumulation_steps,
    )

    if cfg.dreambooth.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)
    if not cfg.dreambooth.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.dreambooth.gradient_accumulation_steps)
    if overrode_max_train_steps:
        cfg.dreambooth.max_train_steps = cfg.dreambooth.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    cfg.dreambooth.num_train_epochs = math.ceil(cfg.dreambooth.max_train_steps / num_update_steps_per_epoch)


    # Train!
    total_batch_size = cfg.dreambooth.train_batch_size * accelerator.num_processes * cfg.dreambooth.gradient_accumulation_steps

    logger.info("***** Running Dreambooth training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {cfg.dreambooth.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg.dreambooth.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {cfg.dreambooth.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(cfg.dreambooth.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0

    for epoch in range(cfg.dreambooth.num_train_epochs):
        unet.train()
        if cfg.dreambooth.train_text_encoder:
            text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict the noise residual
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states)[0]
                noise_pred = noise_pred.sample

                if cfg.dreambooth.with_prior_preservation:
                    # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
                    noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
                    noise, noise_prior = torch.chunk(noise, 2, dim=0)

                    # Compute instance loss
                    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="none").mean([1, 2, 3]).mean()

                    # Compute prior loss
                    prior_loss = F.mse_loss(noise_pred_prior.float(), noise_prior.float(), reduction="mean")
                    # Add the prior loss to the instance loss.
                    loss = loss + cfg.dreambooth.prior_loss_weight * prior_loss
                else:
                    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters(), text_encoder.parameters())
                        if cfg.dreambooth.train_text_encoder
                        else unet.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, cfg.dreambooth.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if accelerator.is_main_process and global_step % 50 == 0:
                logger.info("Ready to save dreambooth model!!!!")
                save_state = {
                    'unet': accelerator.unwrap_model(unet).state_dict(),
                    'encoder': accelerator.unwrap_model(text_encoder).state_dict(),
                }
                logger.info('saving model at {}'.format(
                    os.path.join(cfg.general.save_path, 'dreambooth_{}.ckp'.format(global_step))))
                torch.save(save_state, os.path.join(cfg.general.save_path, 'dreambooth_{}.ckp'.format(global_step)))

            torch.cuda.empty_cache()
            if global_step > cfg.dreambooth.max_train_steps:
                return
        accelerator.wait_for_everyone()

def inference(device, unet, vae, tokenizer, text_encoder, prompt, cfg, logger):
    vae.eval()
    unet.eval()
    text_encoder.eval()

    uncond_input = tokenizer(
        [""] * cfg.dreambooth.inference_batch_size, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
    input_ids = tokenizer(
            [prompt] * cfg.dreambooth.inference_batch_size,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to(device)
    text_embeddings = torch.cat([uncond_embeddings, text_encoder(input_ids)[0]])

    latents = torch.randn(
        (cfg.dreambooth.inference_batch_size, 4, 64, 64),
    ).to(device)

    noise_scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    noise_scheduler.set_timesteps(51)
    guidance_scale = 7.5
    latents = latents * noise_scheduler.init_noise_sigma

    for index, t in enumerate(tqdm(noise_scheduler.timesteps)):
        with torch.no_grad():
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)

            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)[0]
            noise_pred = noise_pred.sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

    logger.info("save images to {}".format(cfg.general.save_path))

    vae.to(latents.device, dtype=latents.dtype)
    with torch.no_grad():
        latents = 1 / 0.18215 * latents
        image = vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        for idx, pil_image in enumerate(pil_images):
            pil_image.save(os.path.join(cfg.general.save_path, "{}_{}.png".format('_'.join(prompt.split(' ')), idx)))

@hydra.main(version_base=None, config_path="conf", config_name="real_image_editing_config")
def main(cfg: DictConfig):

    cfg.general.save_path = os.path.join(cfg.general.save_path, 'dreambooth')

    if cfg.general.seed is not None:
        set_seed(cfg.general.seed)

    with open(cfg.general.unet_config) as f:
        unet_config = json.load(f)
    # load pretrained models and schedular
    unet = unet_2d_condition.UNet2DConditionModel(**unet_config).from_pretrained(cfg.general.model_path, subfolder="unet")
    tokenizer = CLIPTokenizer.from_pretrained(cfg.general.model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(cfg.general.model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(cfg.general.model_path, subfolder="vae")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mixed_precision = 'fp16' if torch.cuda.is_available() else 'no'
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.text_inversion.gradient_accumulation_steps,
        mixed_precision=mixed_precision
    )

    if not os.path.exists(cfg.general.save_path) and accelerator.is_main_process:
        os.makedirs(cfg.general.save_path)

    logger = setup_logger(cfg.general.save_path, __name__)

    logger.info(cfg)
    # Save cfg
    logger.info("save config to {}".format(os.path.join(cfg.general.save_path, 'config.yaml')))
    OmegaConf.save(cfg, os.path.join(cfg.general.save_path, 'config.yaml'))

    # Move vae and unet to device
    vae.to(device)
    unet.to(device)
    text_encoder.to(device)

    if cfg.dreambooth.text_inversion_path != '':
        logger.info("load text inversion ckp from {}".format(cfg.dreambooth.text_inversion_path))
        text_encoder, tokenizer = load_text_inversion(text_encoder, tokenizer, cfg.text_inversion.placeholder_token, cfg.dreambooth.text_inversion_path)

    if cfg.dreambooth.inference:
        ckp = torch.load(cfg.dreambooth.ckp_path)
        unet.load_state_dict(ckp['unet'])
        text_encoder.load_state_dict(ckp['encoder'])


    if not cfg.dreambooth.inference:
        train_dreambooth(device, unet, vae, tokenizer, text_encoder, cfg, accelerator, logger)

    if cfg.dreambooth.new_prompt != '':
        prompt = cfg.dreambooth.new_prompt.format(cfg.text_inversion.placeholder_token)
    else:
        prompt = cfg.dreambooth.example_prompt.format(cfg.text_inversion.placeholder_token)
    inference(device, unet, vae, tokenizer, text_encoder, prompt, cfg, logger)

if __name__ == "__main__":
    main()


