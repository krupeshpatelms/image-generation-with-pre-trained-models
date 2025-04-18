import torch
from diffusers import StableDiffusionPipeline

# Load Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Define a nature-themed prompt
prompt = "A serene forest with sunlight filtering through the trees, birds chirping, and a clear blue sky"

# Generate the nature image
image = pipe(prompt).images[0]

# Save and display the image
image.save("nature_image.png")
image.show()
