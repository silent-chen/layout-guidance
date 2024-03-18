import argparse
import itertools
import math
import os
import random
from pathlib import Path
from typing import Optional
import hydra
import json
from omegaconf import DictConfig

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
from omegaconf import OmegaConf

import PIL
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, LMSDiscreteScheduler
from my_model import unet_2d_condition
from diffusers.optimization import get_scheduler
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from utils import setup_logger, load_text_inversion
def save_progress(text_encoder, placeholder_token_id, accelerator, iteration_idx, cfg, logger):
    logger.info("Saving embeddings to {}".format(os.path.join(cfg.general.save_path, "learned_embeds_iteration_{}.bin")))
    learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[placeholder_token_id]
    learned_embeds_dict = {cfg.text_inversion.placeholder_token: learned_embeds.detach().cpu()}
    torch.save(learned_embeds_dict, os.path.join(cfg.general.save_path, "learned_embeds_iteration_{}.bin".format(iteration_idx)))

imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]


class TextualInversionDataset(Dataset):
    def __init__(
            self,
            data_root,
            tokenizer,
            learnable_property="object",  # [object, style]
            size=512,
            repeats=100,
            interpolation="bicubic",
            flip_p=0.5,
            set="train",
            placeholder_token="*",
            center_crop=False,
            randaug=False
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p
        self.randaug = randaug
        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "linear": PIL.Image.LINEAR,
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
        }[interpolation]

        self.templates = imagenet_style_templates_small if learnable_property == "style" else imagenet_templates_small
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)
        self.transform = transforms.RandAugment()
        # self.transform = transforms.Compose([
        #     transforms.Resize(int(size * 5/4)),
        #     transforms.CenterCrop(int(size * 5/4)),
        #     transforms.RandomApply([
        #         transforms.RandomRotation(degrees=10, fill=255),
        #         transforms.CenterCrop(int(size * 5/6)),
        #         transforms.Resize(size),
        #     ], p=0.75),
        #     transforms.RandomResizedCrop(size, scale=(0.85, 1.15)),
        #     # transforms.RandomApply([transforms.ColorJitter(0.04, 0.04, 0.04, 0.04)], p=0.75),
        #     # transforms.RandomGrayscale(p=0.10),
        #     transforms.RandomApply([transforms.GaussianBlur(5, (0.1, 2))], p=0.10),
        # ])
    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        text = random.choice(self.templates).format(placeholder_string)

        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)
        if self.randaug:
            print("using randaug")
            image = self.transform(image)
        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example


def unfreeze_params(params):
    for param in params:
        param.requires_grad = True

def freeze_params(params):
    for param in params:
        param.requires_grad = False


def text_inversion(device, unet, vae, tokenizer, text_encoder, cfg, accelerator, logger):
    num_added_tokens = tokenizer.add_tokens(cfg.text_inversion.placeholder_token)
    if num_added_tokens == 0:
        raise ValueError(
            f"The tokenizer already contains the token {cfg.text_inversion.placeholder_token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )

    # Convert the initializer_token, placeholder_token to ids
    token_ids = tokenizer.encode(cfg.text_inversion.initial_token, add_special_tokens=False)
    # Check if initializer_token is a single token or a sequence of tokens
    if len(token_ids) > 1:
        raise ValueError("The initializer token must be a single token.")

    initializer_token_id = token_ids[0]
    placeholder_token_id = tokenizer.convert_tokens_to_ids(cfg.text_inversion.placeholder_token)

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Initialise the newly added placeholder token with the embeddings of the initializer token
    token_embeds = text_encoder.get_input_embeddings().weight.data
    token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]

    # Freeze vae and unet
    freeze_params(vae.parameters())
    freeze_params(unet.parameters())
    # Freeze all parameters except for the token embeddings in text encoder
    params_to_freeze = itertools.chain(
        text_encoder.text_model.encoder.parameters(),
        text_encoder.text_model.final_layer_norm.parameters(),
        text_encoder.text_model.embeddings.position_embedding.parameters(),
    )
    freeze_params(params_to_freeze)

    if cfg.text_inversion.scale_lr:
        cfg.text_inversion.lr = (
                cfg.text_inversion.lr * cfg.text_inversion.gradient_accumulation_steps * cfg.text_inversion.batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
        lr=cfg.text_inversion.lr,
        betas=(cfg.text_inversion.adam_beta1, cfg.text_inversion.adam_beta2),
        weight_decay=cfg.text_inversion.adam_weight_decay,
        eps=cfg.text_inversion.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_config("runwayml/stable-diffusion-v1-5", subfolder="scheduler")

    lr_scheduler = get_scheduler(
        cfg.text_inversion.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.text_inversion.lr_warmup_steps,
        num_training_steps=cfg.text_inversion.max_train_steps
    )
    inverted_image_path = cfg.general.save_path if cfg.text_inversion.image_path == ' ' else cfg.text_inversion.image_path
    logger.info('load image at {}'.format(inverted_image_path))
    train_dataset = TextualInversionDataset(
        data_root=inverted_image_path,
        tokenizer=tokenizer,
        size=cfg.text_inversion.resolution,
        placeholder_token=cfg.text_inversion.placeholder_token,
        repeats=cfg.text_inversion.repeats,
        learnable_property=cfg.text_inversion.learnable_property,
        center_crop=cfg.text_inversion.center_crop,
        set="train",
        randaug=cfg.text_inversion.randaug,
    )
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.text_inversion.batch_size, shuffle=True)

    text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        text_encoder, optimizer, train_dataloader,  lr_scheduler
    )



    # Keep vae and unet in eval model as we don't train these
    vae.eval()
    unet.eval()


    total_batch_size = cfg.text_inversion.batch_size * accelerator.num_processes * cfg.text_inversion.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {cfg.text_inversion.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.text_inversion.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg.text_inversion.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {cfg.text_inversion.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(cfg.text_inversion.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0

    for epoch in range(cfg.text_inversion.num_train_epochs):
        text_encoder.train()
        for step, batch in enumerate(train_dataloader):

            with accelerator.accumulate(text_encoder):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"]).latent_dist.sample().detach()
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn(latents.shape).to(latents.device)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                ).long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict the noise residual
                noise_pred, _, _, _ = unet(noisy_latents, timesteps, encoder_hidden_states)
                noise_pred = noise_pred.sample

                # import pdb; pdb.set_trace()

                loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
                accelerator.backward(loss)

                # Zero out the gradients for all token embeddings except the newly added
                # embeddings for the concept, as we only want to optimize the concept embeddings
                if accelerator.num_processes > 1:
                    grads = text_encoder.module.get_input_embeddings().weight.grad
                else:
                    grads = text_encoder.get_input_embeddings().weight.grad
                # Get the index for tokens that we want to zero the grads for
                index_grads_to_zero = torch.arange(len(tokenizer)) != placeholder_token_id
                grads.data[index_grads_to_zero, :] = grads.data[index_grads_to_zero, :].fill_(0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if global_step % cfg.text_inversion.save_steps == 0:
                    save_progress(text_encoder, placeholder_token_id, accelerator, global_step, cfg, logger)

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            # accelerator.log(logs, step=global_step)

            if global_step >= cfg.text_inversion.max_train_steps:
                logger.info("reach the maximum iteration")
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

    cfg.general.save_path = os.path.join(cfg.general.save_path, 'text_inversion')

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


    # Move models to device
    vae.to(accelerator.device)
    unet.to(accelerator.device)
    text_encoder.to(accelerator.device)

    if not cfg.text_inversion.inference:
        text_inversion(device, unet, vae, tokenizer, text_encoder, cfg, accelerator, logger)
    else:
        text_encoder, tokenizer = load_text_inversion(text_encoder, tokenizer, cfg.text_inversion.placeholder_token, cfg.text_inversion.embedding_ckp)

    if cfg.text_inversion.new_prompt != '':
        prompt = cfg.text_inversion.new_prompt.format(cfg.text_inversion.placeholder_token)
    else:
        prompt = cfg.text_inversion.example_prompt.format(cfg.text_inversion.placeholder_token)
    inference(device, unet, vae, tokenizer, text_encoder, prompt, cfg, logger)





if __name__ == "__main__":
    main()