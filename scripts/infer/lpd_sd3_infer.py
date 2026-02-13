import os
import sys

# BOOTSTRAP_PATHS: allow running scripts directly without installing the package.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained("//zoo/stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

image = pipe(
    "In this image we can see a bowl of broccoli, a bowl of ladyfinger (okra), a bowl of leafy vegetable, a container with cabbage in the upper right part of the image, a metallic kettle with a silver color, made of metal, round in shape, featuring a handle, a lid, and a shiny surface in the upper part of the image, a wooden spoon made of wooden material with an elongated shape in the upper left part of the image, and a plate with a spoon are placed on a table. The background includes a kitchen setting with a stove and a kettle, suggesting a cooking environment. The image is brightly lit with soft, even lighting, suggesting an indoor setting with ample ambient light, possibly from overhead fixtures. The light source appears to be positioned above and slightly in front of the scene, minimizing harsh shadows. The style of the image is a realistic photo. The bowl of broccoli is placed to the left of the bowl of ladyfinger (okra). The bowl of leafy vegetable is positioned to the right of the bowl of ladyfinger (okra). The container with cabbage is located above the bowl of leafy vegetable. The wooden spoon is placed on the plate with a spoon, which is to the left of the metallic kettle. The metallic kettle is situated above the bowl of ladyfinger (okra).",
    num_inference_steps=40,
    max_sequence_length=512,
    guidance_scale=4.5,
).images[0]
image.save("capybara.png")
