import cv2
import numpy as np
import torch
from PIL import ImageOps
from PIL import Image


def image_to_tensor(input):
    i = ImageOps.exif_transpose(input)
    image = i.convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(image)[None,]
    return tensor


def image_to_np(input):
    i = ImageOps.exif_transpose(input)
    image = i.convert("RGB")
    image_np = np.array(image).astype(np.uint8)
    return image_np


def tensor_to_np(image):
    image = image[0]
    i = 255. * image.cpu().numpy()
    result = np.clip(i, 0, 255).astype(np.uint8)
    return result


def img_to_mask(input):
    i = ImageOps.exif_transpose(input)
    image = i.convert("RGB")
    new_np = np.array(image).astype(np.float32) / 255.0
    mask_tensor = torch.from_numpy(new_np).permute(2, 0, 1)[0:1, :, :]
    return mask_tensor


def image_np_to_image_tensor(input):
    image = input.astype(np.float32) / 255.0
    tensor = torch.from_numpy(image)[None,]
    return tensor


def mask_np2_to_mask_tensor(input):
    image = input.astype(np.float32)
    tensor = torch.from_numpy(image)[None,]
    return tensor


def mask_np3_to_mask_tensor(input):
    image = input.astype(np.float32)
    tensor = torch.from_numpy(image).permute(2, 0, 1)[0:1, :, :]
    return tensor


def mask_tensor_to_mask_np3(input):
    result = input.permute(1, 2, 0).cpu().numpy()
    return result


def tensor_to_img(image):
    image = image[0]
    i = 255. * image.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8)).convert("RGB")
    return img



def image_np_to_mask(input):
    new_np = input.astype(np.float32) / 255.0
    tensor = torch.from_numpy(new_np).permute(2, 0, 1)[0:1, :, :]
    return tensor
