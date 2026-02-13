import os
import sys

# BOOTSTRAP_PATHS: allow running scripts directly without installing the package.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import pandas as pd
import torch
import torch.nn as nn

from transformers import T5Tokenizer, T5EncoderModel, AutoModel
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers import AutoPipelineForText2Image

import os
import json
import pandas as pd
from tqdm import tqdm
import random
from PIL import Image

prompts = [
'''The image portrays a solitary knight in full plate armor, captured from a low-angle perspective that emphasizes his heroic stature. The armor is forged from burnished, dark steel, its surface intricately etched with silver filigree along the edges of the pauldrons and greaves. A single, glowing rune is visible on the center of the chest plate, casting a faint, ethereal light. The knight stands in a stoic pose on a windswept, rocky precipice, his gaze directed towards a colossal castle looming in the background. A heavy, crimson cape, its texture appearing weathered and frayed at the hem, billows dramatically behind him. The castle is a formidable gothic structure, constructed from dark, moss-covered stone that suggests great age. Its towering spires and crenelated walls are silhouetted against a tumultuous sky filled with swirling, dark gray storm clouds. A dramatic shaft of golden sunlight breaks through the cloud cover, illuminating the knight and creating a high-contrast effect with long, sharp shadows. The foreground is composed of jagged rock and sparse, hardy grass. The overall color palette is muted, dominated by shades of gray and deep blue, punctuated by the stark crimson of the cape and the golden light. The style is that of hyper-realistic digital concept art, with a focus on the textural details of the metal, stone, and fabric, creating a dramatic and epic fantasy scene.''',
'''A majestic German Shepherd with strong features, is depicted in a full-body pose sitting attentively. The entire form of the dog is rendered in a vibrant, intricate stained glass style. Each segment of its body is a separate piece of colored glass, outlined with thick, dark lines that mimic the lead strips in a real stained glass window. The colors are rich and varied, with deep ambers, brilliant golds, and earthy browns for its coat, and perhaps some striking blues or greens in the eyes to draw focus. The light source seems to be coming from behind the dog, causing the colors to glow with an inner luminescence. The texture of the glass is not uniform; some pieces are smooth and translucent, while others have a more textured, almost liquid appearance, creating a dynamic interplay of light and color across the dog's form. The focus is entirely on the dog, with no background detail to detract from the intricate glasswork of its body.''',
'''A hyper-detailed, macro shot of a human eye, presented not as an organ of sight, but as a gateway to a lost world of intricate craftsmanship. The iris is a masterfully crafted, antique horological mechanism, a complex universe of miniature, interlocking gears and cogs made from polished brass, copper, and tarnished silver. Each metallic piece is exquisitely detailed, with tiny, functional teeth that seem to pulse with a slow, rhythmic, and almost imperceptible life. The vibrant color of the iris is replaced by the warm, metallic sheen of the gears, with ruby and sapphire jewels embedded as tiny, gleaming pivots. At the center, the pupil is not a void but the deep, dark face of a miniature clock, its impossibly thin, filigreed hands frozen at a moment of profound significance. The delicate, thread-like veins in the sclera are reimagined as fine, coiling copper wires, connecting the central mechanism to the unseen power source at the edge of the frame. The entire piece is captured under a soft, focused light that highlights the metallic textures and casts deep, dramatic shadows within the complex machinery, suggesting immense depth. The background is a stark, velvety black, ensuring nothing distracts from the mesmerizing, mechanical soul of the eye.''',
'''A formidable knight stands in a powerful, regal pose, not forged in the fires of a smithy, but grown from the heart of an ancient, mystical forest. The knight's entire suit of armor is composed of living, enchanted wood, a seamless blend of natural defense and elegant design. The breastplate is sculpted from the dense, unyielding heartwood of an ironwood tree, its surface a tapestry of swirling grain and deep, textured bark that has been polished to a low, earthy luster. Intricate patterns of phosphorescent fungi and glowing moss creep along the crevices and joints of the armor, tracing the contours of the pauldrons, gauntlets, and greaves with a soft, ethereal light in shades of vibrant emerald and cool sapphire. The helmet is carved from a single, massive burl of an ancient oak, its form both protective and organic, with a visor made of tightly woven, thorny vines that conceals the knight's face entirely. From behind this natural grille, a soft, warm light emanates, suggesting a being of pure, natural energy within. The knight's sword is a single, massive thorn of petrified wood, impossibly sharp and infused with a faint, magical aura. The background is a simple, deep, and uniform black, serving to highlight the intricate details and the soft, natural glow of this woodland protector.''',
'''The image presents a 3D rendering of a horse, captured in a profile view. The horse is depicted \
in a state of motion, with its mane and tail flowing behind it. The horse's body is composed \
of a network of lines and curves, suggesting a complex mechanical structure. This intricate \
design is further emphasized by the presence of gears and other mechanical components, which \
are integrated into the horse's body. The background of the image is a dark blue, providing a \
stark contrast to the horse and its mechanical components. The overall composition of the image \
suggests a blend of organic and mechanical elements, creating a unique and intriguing visual.''',
'''The image is a surrealist, photorealistic close-up of a human eye, rendered with a fantastical and ethereal quality. The iris is a mesmerizing, deep sapphire blue, with intricate, swirling patterns of silver and gold that seem to shift and catch the light. Within the glossy, reflective pupil, a miniature galaxy of distant stars and nebulae is mirrored. Long, delicate eyelashes, dusted with a fine, shimmering silver powder, curl upwards, each strand individually defined. The eye is framed by a lush, cascading wreath of bioluminescent flowers in full, radiant bloom. Predominantly in shades of cosmic blue, royal purple, and deep indigo, the arrangement includes luminous irises with velvety petals, delicate lavender sprigs that emit a soft glow, and pansies with faces like miniature galaxies. Glistening dewdrops cling to the petals, each one reflecting the starlit scene in the eye's pupil. The background is a deep, velvety indigo night sky, filled with a dense field of twinkling stars and faint, ethereal wisps of a purple nebula. The primary light source emanates from the flowers themselves, casting a soft, magical glow that illuminates the intricate details of the eye and creates a captivating, high-contrast image. The overall style is that of a hyper-detailed digital painting, blending realism with fantasy elements to evoke a sense of wonder and natural beauty.''',
'''A vibrant hummingbird, a jewel of nature, is captured mid-hover, its form a masterpiece of photorealistic detail. The bird's tiny body is a whirlwind of motion, its wings beating so rapidly they are rendered as a translucent, shimmering blur against the soft-focus background of lush, green foliage. Every feather is meticulously defined, from the iridescent emerald and sapphire plumage on its back to the downy white of its underbelly. Sunlight catches the metallic sheen of its feathers, creating a dazzling play of light and color that shifts with every imperceptible movement. Its long, delicate beak, a needle-thin instrument of precision, is poised just before a flower, though the flower itself remains indistinct. The bird's eye, a tiny bead of polished obsidian, is sharp and intelligent, a focal point of life and energy in the composition. The overall atmosphere is one of vibrant life and ephemeral beauty, a fleeting moment of natural perfection frozen in time. The bright, high-key lighting enhances the scene's realism and imbues it with a sense of joy and vitality. The background, a soft, creamy bokeh of indistinguishable leaves and sunlight, serves to isolate the hummingbird, making it the undisputed star of the image.''',
'''A sleek, enigmatic feline, a cat of indeterminate breed, is the central figure, poised in a state of serene contemplation. Its body is not of flesh and bone, but meticulously sculpted from a complex lattice of polished, interlocking obsidian shards. Each piece is perfectly fitted against the next, creating a mosaic of deep, lustrous black that absorbs the light. The cat's form is defined by the sharp, clean edges of these volcanic glass fragments, giving its natural curves a subtle, geometric undertone. Glimmering veins of molten gold run through the cracks between the shards, glowing with a soft, internal heat that pulses rhythmically, like a slow heartbeat. These golden rivers trace the contours of the cat's muscles and skeleton, outlining its elegant spine, the delicate structure of its paws, and the graceful curve of its tail. Its eyes are two brilliant, round-cut rubies, catching an unseen light source and casting a faint, crimson glow. The whiskers are impossibly thin strands of spun platinum, fanning out from its muzzle with metallic precision. The entire figure rests upon a simple, unadorned, and dimly lit surface, ensuring that all focus remains on the cat's extraordinary construction—a masterful fusion of natural grace and exquisite, dark craftsmanship.''',
'''The image portrays a female character with a fantasy-inspired design. She has long, dark hair \
that cascades down her shoulders. Her skin is pale, and her eyes are a striking shade of blue. \
The character's face is adorned with intricate gold and pink makeup, which includes elaborate \
patterns and designs around her eyes and on her cheeks. Atop her head, she wears a crown made \
of gold and pink roses, with the roses arranged in a circular pattern. The crown is detailed, with \
each rose appearing to have a glossy finish. The character's attire consists of a gold and pink dress \
that is embellished with what appears to be feathers or leaves, adding to the fantasy aesthetic. The \
background of the image is dark, which contrasts with the character's pale skin and the bright \
colors of her makeup and attire. The lighting in the image highlights the character's features \
and the details of her makeup and attire, creating a dramatic and captivating effect. There are no \
visible texts or brands in the image. The style of the image is highly stylized and artistic, with \
a focus on the character's beauty and the intricate details of her makeup and attire. The image \
is likely a digital artwork or a concept illustration, given the level of detail and the fantastical \
elements present.''',
'''The image captures a scene of a large, modern building perched on a cliff. The building, painted \
in shades of blue and gray, stands out against the backdrop of a cloudy sky. The cliff itself is \
a mix of dirt and grass, adding a touch of nature to the otherwise man-made structure. In the \
foreground, a group of people can be seen walking along a path that leads up to the building. \
Their presence adds a sense of scale to the image, highlighting the grandeur of the building. The \
sky above is filled with clouds, casting a soft, diffused light over the scene. This light enhances \
the colors of the building and the surrounding landscape, creating a visually striking image. \
Overall, the image presents a harmonious blend of architecture and nature, with the modern \
building seamlessly integrated into the natural landscape.''',
'''A magnificent jellyfish, a creature of ethereal beauty, commands the center of the frame, captured in a moment of serene, balletic grace. Its bell, a perfect, translucent dome, is a marvel of natural architecture, rendered with hyper-realistic detail. Through its glassy surface, the intricate, labyrinthine network of its internal structures is faintly visible, a delicate filigree of soft pinks and purples. The surface of the bell catches and refracts the light, creating a dazzling, iridescent sheen that shifts with every subtle movement. From the bell's lower edge, a cascade of tentacles descends, a symphony of color and form. Some are long and trailing, like silken ribbons of neon pink and electric blue, while others are shorter and frilled, a delicate, lacy curtain of vibrant orange and sunshine yellow. The entire creature is imbued with a gentle, bioluminescent glow, a soft, internal light that seems to pulse with a life of its own. The jellyfish is set against a backdrop of the deep, cerulean sea, the water so clear that the sunlight from above penetrates its depths, creating a brilliant, sun-drenched environment. In the lower corners of the frame, vibrant coral formations, in shades of fiery red and deep violet, add a touch of contrasting color and texture, grounding the ethereal jellyfish in a thriving, underwater ecosystem.''',
'''A close-up, almost intimate, shot of a knight's helm, but it is forged not from steel, but from the very fabric of a captured nebula. The entire helmet swirls with the deep indigos, magentas, and cyans of a distant galaxy, with miniature stars igniting and dying within its cosmic-spun material. The visor is a sheet of pure, polished obsidian, so dark it seems to drink the light, and behind it, two points of intense, white-hot starlight burn with a steady, unwavering gaze, hinting at the consciousness within. The helmet’s crest is not of feather or metal, but a standing wave of solidified light, a blade of pure energy that cuts through the dimness. Light from an unseen source catches on the helmet’s contours, not with a metallic sheen, but by causing the internal galaxies to glow brighter, the nebulae to churn, and the star-fire to pulse with a slow, silent rhythm. The surface isn't smooth but has a subtle, crystalline texture, as if space itself has been faceted and polished. The background is a simple, deep black, a void that serves only to emphasize the celestial majesty of the figure, making the knight appear as a solitary constellation in the vast emptiness of space.''',
'''A magnificent castle, seemingly carved from a single, colossal amethyst, stands in silent grandeur. Its towering spires and crenelated walls are not constructed from stone but are instead faceted and polished surfaces of the deep purple gemstone. Light from an unseen source refracts through the crystalline structure, creating a mesmerizing internal luminescence that pulses with a soft, violet glow. The castle's architecture is both familiar and fantastical, with classic medieval towers and archways rendered in the sharp, geometric lines of a cut gem. Intricate filigree patterns, like frozen lightning, are etched into the amethyst, their silver-white lines glowing with a faint, ethereal light. These patterns trace the contours of the castle, defining its gates, windows, and the delicate tracery of its highest towers. The drawbridge is a solid sheet of polished quartz, its transparent surface revealing the shimmering, crystalline depths below. The entire structure rests on a smooth, dark, and reflective surface, creating a perfect, mirrored image of the glowing amethyst castle against an endless, dark void. This masterful creation is a breathtaking fusion of formidable fortification and delicate, crystalline beauty, a fortress of light and shadow.''',
]
pipe = StableDiffusionPipeline.from_pretrained("//zoo/stabilityai/stable-diffusion-2-1-base", 
                                               safety_checker=None,
                                               torch_dtype=torch.float16).to("cuda")
for step, prompt in enumerate(prompts):
    generator = torch.Generator("cuda").manual_seed(43)
    with torch.no_grad():
        image = pipe(
            prompt,
            generator=generator
        ).images[0]
        image.save(f"{step}.png")
exit()
pipe.set_progress_bar_config(disable=True)

save_dir = "//samples/long_PDD/sd2/dm_bench/sd1"
os.makedirs(save_dir, exist_ok=True)
out_meta = []
with open('//data-juicer/DetailMaster_Dataset/DetailMaster_Dataset.json', 'r') as file:
    meta = json.load(file)
progress_bar = tqdm(
    range(0, len(meta)),
    initial=0,
    desc="Steps",
)
for step, batch in enumerate(meta):
    generator = torch.Generator("cuda").manual_seed(3467)
    progress_bar.update(1)
    prompt = batch['polished_prompt']
    out_meta.append({"output_image_name": f"{step}.png", "image_id": f"{batch['dataset_target']}_{batch['image_id']}"})
    if os.path.isfile(f"{save_dir}/{step}.png"):
        continue
    with torch.no_grad():
        image = pipe(
            prompt,
            generator=generator
        ).images[0]
        image.save(f"{save_dir}/{step}.png")
with open(f"{save_dir}.json", "w") as f:
    json.dump(out_meta, f, indent=4)