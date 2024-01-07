import numpy as np
from PIL import Image


def tensors_to_imgs(tensors):
    images = []
    for image in tensors:
        i = 255. * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8)).convert("RGB")
        images.append(image)
    return images
