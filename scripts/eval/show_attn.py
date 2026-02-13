import os
import sys

# BOOTSTRAP_PATHS: allow running scripts directly without installing the package.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from PIL import Image

attn_map = np.transpose(np.load("attention_map_tokens_5.npy"))

# attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
# attn_map = np.clip(attn_map/25, 0.0, 1.0)

out_map = []
for am in attn_map:
    # am = am / am.sum()
    am = (am - am.min()) / (am.max() - am.min())
    # out_m = np.clip(out_m, 0.0, 1.0)
    print(am.min(), am.max(), am.shape)
    # out_m =  (out_m - out_m.min()) / (out_m.max() - out_m.min())
    out_map.append(am)
attn_map = np.stack(out_map)

# attn_map = np.repeat(np.repeat(attn_map, 41, axis=0), 41, axis=1)
colormap = colormaps['viridis']
attention_map_colored = colormap(attn_map)

# 3. Convert to an 8-bit integer RGB array for PIL
# PIL's Image.fromarray function works best with uint8 data.
# We scale the [0, 1] float values to [0, 255] integers.
# We also slice off the alpha channel with `[:, :, :3]` to get just RGB.
image_array = (attention_map_colored[:, :, :3] * 255).astype(np.uint8)

# 4. Create a PIL Image from the NumPy array
pil_image = Image.fromarray(image_array)

pil_image.save('attn_map.jpg')

# plt.figure(figsize=(16, 6))
# plt.imshow(
#     out_map,
#     cmap='viridis',    # Use a good colormap
#     aspect='auto'      # Fill the rectangular figure shape
# )
# plt.colorbar(label='Attention Weight')
# plt.title('Attention Map: 64 Query Tokens to 576 Key/Value Tokens')
# plt.xlabel('Key/Value Tokens (576)')
# plt.ylabel('Query Tokens (64)')
# plt.savefig('attn_map.jpg', bbox_inches='tight', dpi=300)

# # It's good practice to close the figure to free up memory
# plt.close()