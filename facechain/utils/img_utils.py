import numpy as np
from PIL import Image

def crop_bottom(pil_file, width):
    if width == 512:
        height = 768
    else:
        height = 1152
    w, h = pil_file.size
    factor = w / width
    new_h = int(h / factor)
    pil_file = pil_file.resize((width, new_h))
    crop_h = min(int(new_h / 32) * 32, height)
    array_file = np.array(pil_file)
    array_file = array_file[:crop_h, :, :]
    output_file = Image.fromarray(array_file)
    return output_file
