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
import cv2
from PIL import Image
import numpy as np
from diffusers import DDPMScheduler
from torchvision import transforms


def pad_and_shear_image(image, pad_width, shear_x=0, shear_y=0, border_color=(255, 255, 255)):
    """
    Pad and shear an image while ensuring content starts from upper-left corner
    
    Args:
        image: Input image (numpy array)
        pad_width: Padding width for all sides
        shear_x: Horizontal shear factor
        shear_y: Vertical shear factor
        border_color: Color for padding and empty areas
    
    Returns:
        Transformed image with content properly positioned
    """
    h, w = image.shape[:2]
    
    # Step 1: Add initial padding
    padded_image = cv2.copyMakeBorder(
        image, pad_width, pad_width, pad_width, pad_width, 
        cv2.BORDER_CONSTANT, value=border_color
    )
    
    padded_h, padded_w = padded_image.shape[:2]
    
    # Step 2: Calculate the bounds after shearing to determine output size
    # Define the four corners of the padded image
    corners = np.array([
        [0, 0, 1],                    # top-left
        [padded_w, 0, 1],            # top-right  
        [0, padded_h, 1],            # bottom-left
        [padded_w, padded_h, 1]      # bottom-right
    ]).T
    
    # Create shear transformation matrix
    shear_matrix = np.array([
        [1, shear_x, 0],
        [shear_y, 1, 0],
        [0, 0, 1]
    ])
    
    # Transform corners to find bounds
    transformed_corners = shear_matrix @ corners
    
    # Find the bounding box of transformed corners
    min_x = np.min(transformed_corners[0, :])
    max_x = np.max(transformed_corners[0, :])
    min_y = np.min(transformed_corners[1, :])
    max_y = np.max(transformed_corners[1, :])
    
    # Calculate output dimensions and translation needed
    output_w = int(np.ceil(max_x - min_x))
    output_h = int(np.ceil(max_y - min_y))
    
    # Translation to ensure content starts from (0,0)
    tx = -min_x
    ty = -min_y
    
    # Create final transformation matrix with translation
    final_matrix = np.float32([
        [1, shear_x, tx],
        [shear_y, 1, ty]
    ])
    
    # Apply transformation
    result = cv2.warpAffine(
        padded_image, final_matrix, (output_w, output_h),
        borderValue=border_color
    )
    
    return result

# noise_scheduler = DDPMScheduler.from_pretrained("//zoo/runwayml/stable-diffusion-v1-5", subfolder="scheduler")
# # OpenCV to PyTorch
# image_rgb = cv2.cvtColor(cv2.imread('//samples/long_PDD/sd2/full/4_component_T64_ct5ELLA-1024-8-4_D10_43/6000_4.png'), cv2.COLOR_BGR2RGB)
# tensor = transforms.ToTensor()(image_rgb) * 2 - 1  # normalizes to [-1,1]
# tensor = tensor.unsqueeze(0)
# noise = torch.randn_like(tensor)
# noisy_tensor = noise_scheduler.add_noise(tensor, noise, torch.tensor((699,), dtype=torch.long))
# noisy_tensor = (noisy_tensor+1)/2
# noisy_tensor = torch.clamp(noisy_tensor, 0, 1)
# noisy_tensor = noisy_tensor.squeeze()
# # PyTorch back to OpenCV
# image_back = (transforms.ToPILImage()(noisy_tensor))
# image_bgr_back = cv2.cvtColor(np.array(image_back), cv2.COLOR_RGB2BGR)
# result = pad_and_shear_image(image_bgr_back, pad_width=10, shear_x=0.0, shear_y=0.15)
# result = result[10:602, 10:522, :]
# cv2.imwrite('input.png', result, params=None)

img_width=512
img_height=512
img = Image.open('//samples/long_PDD/sd2/full/4_component_T64_ct5ELLA-1024-8-4_D10_43/6000_7_components.png')
img1 = img.crop((0, 0, img_width, img_height))

# Second image: x=512 to x=1024
img2 = img.crop((img_width, 0, img_width * 2, img_height))

# Third image: x=1024 to x=1536
img3 = img.crop((img_width * 2, 0, img_width * 3, img_height))

# Fourth image (rightmost): x=1536 to x=2048
img4 = img.crop((img_width * 3, 0, img_width * 4, img_height))

# 3) Create new image with first and fourth swapped
# New order: img4, img2, img3, img1 (fourth, second, third, first)
new_img = Image.new(img.mode, (2048, 512))

# Paste images in new positions
new_img.paste(img2, (0, 0))                    # Fourth image goes to first position
new_img.paste(img3, (img_width, 0))           # Second image stays in second position
new_img.paste(img1, (img_width * 2, 0))       # Third image stays in third position
new_img.paste(img4, (img_width * 3, 0))
new_img.save('components.png')