# This script generates a nature-themed image using Stable Diffusion

import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

prompt = "A peaceful waterfall in a lush green forest, birds flying around"
image = pipe(prompt).images[0]

image.save("nature_image.png")
print("Nature image generated and saved as nature_image.png")
