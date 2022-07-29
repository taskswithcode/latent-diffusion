import argparse, os, sys, datetime, glob, importlib, csv
import sys
import yaml
import torch
from omegaconf import OmegaConf
import pdb
from taming.models.vqgan import VQModel, GumbelVQ
import io
import os, sys
import requests
import PIL
from PIL import Image
from PIL import ImageDraw, ImageFont
import numpy as np
import argparse
from einops import rearrange

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from dall_e          import map_pixels, unmap_pixels, load_model
from IPython.display import display, display_markdown
from ldm.util import instantiate_from_config

def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config

def download_image(url):
    resp = requests.get(url)
    resp.raise_for_status()
    return PIL.Image.open(io.BytesIO(resp.content))


def preprocess(img, target_image_size=256, map_dalle=True):
    s = min(img.size)
    
    if s < target_image_size:
        raise ValueError(f'min dim for image {s} < {target_image_size}')
        
    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    if map_dalle: 
      img = map_pixels(img)
    return img

def preprocess_vqgan(x):
  x = 2.*x - 1.
  return x

def custom_to_pil(x):
  x = x.detach().cpu()
  x = torch.clamp(x, -1., 1.)
  x = (x + 1.)/2.
  x = x.permute(1,2,0).numpy()
  x = (255*x).astype(np.uint8)
  x = Image.fromarray(x)
  if not x.mode == "RGB":
    x = x.convert("RGB")
  return x

def reconstruct_with_vqgan(x, model):
  #pdb.set_trace()
  #z = rearrange(x, 'a b c d -> b c d a').contiguous()
  # could also use model(x) for reconstruction but use explicit encoding and decoding here
  #z, _, [_, _, indices] = model.encode(x)
  #pdb.set_trace()
  pdb.set_trace()  
  #x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
  posterior = model.encode(x)
  z = posterior.mode()
  #print(f"VQGAN --- {model.__class__.__name__}: latent shape: {z.shape[2:]}")
  xrec = model.decode(z)
  return xrec

def stack_reconstructions(input, x0):
  titles=["Input", "KL-f4"]

  font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf", 22)
  assert input.size == x0.size
  w, h = input.size[0], input.size[1]
  img = Image.new("RGB", (7*w, h))
  img.paste(input, (0,0))
  img.paste(x0, (1*w,0))
  for i, title in enumerate(titles):
    ImageDraw.Draw(img).text((i*w, 0), f'{title}', (255, 255, 255), font=font) # coordinates, text, color, font
  return img


def reconstruction_pipeline(model,url,device, size=320,is_local=False):
    if (is_local):
        x_vqgan = preprocess(PIL.Image.open(url), target_image_size=size, map_dalle=False)
    else:
        x_vqgan = preprocess(download_image(url), target_image_size=size, map_dalle=False)
    x_vqgan = x_vqgan.to(device)
    print(f"input is of size: {x_vqgan.shape}")
    x0 = reconstruct_with_vqgan(preprocess_vqgan(x_vqgan), model)
    img = stack_reconstructions(custom_to_pil(preprocess_vqgan(x_vqgan[0])),custom_to_pil(x0[0]))
    #img = stack_reconstructions(custom_to_pil(x_vqgan[0]),custom_to_pil(x0[0]))
    return img

if __name__ == "__main__":
    DEVICE = torch.device("cpu")
    sys.path.append(os.getcwd())
    config = load_config("models/first_stage_models/kl-f4/config.yaml", display=False)
    model = instantiate_from_config(config.model)
    #img = reconstruction_pipeline(model,url='https://heibox.uni-heidelberg.de/f/7bb608381aae4539ba7a/?dl=1',device=DEVICE, size=384,is_local=False)
    #img = reconstruction_pipeline(model,url='https://heibox.uni-heidelberg.de/f/be6f4ff34e1544109563/?dl=1',device=DEVICE, size=384,is_local=False)
    img = reconstruction_pipeline(model,url='https://heibox.uni-heidelberg.de/f/e41f5053cbd34f11a8d5/?dl=1',device=DEVICE, size=384,is_local=False)
    img.save("kloutput19.png")

