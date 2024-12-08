# InternVL2_5-1B



!pip install -U bitsandbytes -q

!pip install decord



import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


# Load the model and tokenizer
path = "OpenGVLab/InternVL2_5-1B"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    load_in_8bit=True,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True
).eval()  # Remove .cuda() here

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)




import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

# Define build_transform and dynamic_preprocess functions
def build_transform(input_size=448):
    """Builds a transformation pipeline for image preprocessing."""
    transform = T.Compose([
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform


def dynamic_preprocess(image, image_size=448, use_thumbnail=True, max_num=12):
    """Dynamically preprocesses the image."""
    # Placeholder for dynamic preprocessing logic (replace with actual implementation)
    # This example simply returns the original image in a list
    return [image]

# Load the model and tokenizer
path = "OpenGVLab/InternVL2_5-1B"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    load_in_8bit=True,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True
).eval()  # Remove .cuda() here

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)


def load_image(image_file, input_size=448, max_num=None):  # Parameter max_num set to None.
    """Loads an image and preprocesses it.
    If 'max_num' is not None, it is passed to dynamic_preprocess, otherwise a default of 12 is used."""

    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    if max_num:  # Check if max_num is provided
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    else:
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=12) # Default max_num to 12
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

image_path = '/content/1.png'
def load_image(image_file, input_size=448, max_num=None):  # Parameter max_num set to None.
    """Loads an image and preprocesses it.
    If 'max_num' is not None, it is passed to dynamic_preprocess, otherwise a default of 12 is used."""

    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    if max_num:  # Check if max_num is provided
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    else:
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=12) # Default max_num to 12
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

pixel_values = load_image(image_path).to(torch.bfloat16) # Keep pixel_values on CPU
generation_config = dict(max_new_tokens=1024, do_sample=True)

question = '<image>\nPlease describe the image shortly.'
response = model.chat(tokenizer, pixel_values, question, generation_config)
print(f'User: {question}\nAssistant: {response}')
