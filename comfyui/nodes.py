# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import os

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel, \
    UniPCMultistepScheduler
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from torch import multiprocessing
from transformers import pipeline as tpipeline

# import pydevd_pycharm
# pydevd_pycharm.settrace('49.7.62.197', port=10090, stdoutToServer=True, stderrToServer=True)

from .model_holder import *
from .utils.img_utils import *

class FCLoraMerge:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "merge_lora_first": ("MODEL",),
                "merge_lora_second": ("MODEL",),
                "multiplier": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.1})
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("lora",)
    FUNCTION = "merge_lora"
    CATEGORY = "facechain/lora"

    def retain_face(self, merge_lora_first, merge_lora_second):
        # pipe = StableDiffusionPipeline.from_pretrained(base_model_path, safety_checker=None, torch_dtype=torch.float32)
        # merge_lora()
        return ()

class FCLoraStyle:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "merge_lora_first": ("MODEL",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("style_lora",)
    FUNCTION = "lora_style"
    CATEGORY = "facechain/lora"

    def lora_style(self, image):
        return (image,)

class FCFaceFusion:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "fusion_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image_face_fusion"
    CATEGORY = "facechain/model"

    def image_face_fusion(self, source_image, fusion_image):
        source_image = tensor_to_img(source_image)
        fusion_image = tensor_to_img(fusion_image)
        result_image = get_image_face_fusion()(dict(template=source_image, user=fusion_image))[OutputKeys.OUTPUT_IMG]
        result_image = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        return (img_to_tensor(result_image),)

class FCFaceDetection:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "face_index": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1})
            }
        }

    RETURN_TYPES = ("IMAGE", "BOX",)
    FUNCTION = "face_detection"
    CATEGORY = "facechain/model"

    def face_detection(self, source_image, face_index):
        pil_source = tensor_to_img(source_image)
        result_dec = get_face_detection()(pil_source)
        keypoints = result_dec['keypoints']
        boxes = result_dec['boxes']
        scores = result_dec['scores']
        keypoint = keypoints[face_index]
        score = scores[face_index]
        box = boxes[face_index]
        box = np.array(box, np.int32)
        crop_result = source_image[:, box[1]:box[3], box[0]:box[2], :]
        return (crop_result, box)

class FCCropMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "face_box": ("BOX",)
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    FUNCTION = "crop_mask"
    CATEGORY = "facechain/mask"

    def crop_mask(self, image, face_box):
        image = tensor_to_np(image)
        inpaint_img_large = image
        mask_large = np.ones_like(inpaint_img_large)
        mask_large1 = np.zeros_like(inpaint_img_large)
        h, w, _ = inpaint_img_large.shape
        face_ratio = 0.45
        cropl = int(max(face_box[3] - face_box[1], face_box[2] - face_box[0]) / face_ratio / 2)
        cx = int((face_box[2] + face_box[0]) / 2)
        cy = int((face_box[1] + face_box[3]) / 2)
        cropup = min(cy, cropl)
        cropbo = min(h - cy, cropl)
        crople = min(cx, cropl)
        cropri = min(w - cx, cropl)
        inpaint_img = np.pad(inpaint_img_large[cy - cropup:cy + cropbo, cx - crople:cx + cropri], ((cropl - cropup, cropl - cropbo), (cropl - crople, cropl - cropri), (0, 0)), 'constant')
        inpaint_img = cv2.resize(inpaint_img, (512, 512))
        inpaint_img = Image.fromarray(cv2.cvtColor(inpaint_img[:, :, ::-1], cv2.COLOR_BGR2RGB))
        mask_large1[cy - cropup:cy + cropbo, cx - crople:cx + cropri] = 1
        mask_large = mask_large * mask_large1
        return (img_to_tensor(inpaint_img), np_to_mask(mask_large))

class FCSegment:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "fc_segment"
    CATEGORY = "facechain/model"

    def segment(segmentation_pipeline, img, ksize=0, eyeh=0, ksize1=0, include_neck=False, warp_mask=None, return_human=False):
        if True:
            result = segmentation_pipeline(img)
            masks = result['masks']
            scores = result['scores']
            labels = result['labels']
            if len(masks) == 0:
                return
            h, w = masks[0].shape
            mask_face = np.zeros((h, w))
            mask_hair = np.zeros((h, w))
            mask_neck = np.zeros((h, w))
            mask_cloth = np.zeros((h, w))
            mask_human = np.zeros((h, w))
            for i in range(len(labels)):
                if scores[i] > 0.8:
                    if labels[i] == 'Torso-skin':
                        mask_neck += masks[i]
                    elif labels[i] == 'Face':
                        mask_face += masks[i]
                    elif labels[i] == 'Human':
                        mask_human += masks[i]
                    elif labels[i] == 'Hair':
                        mask_hair += masks[i]
                    elif labels[i] == 'UpperClothes' or labels[i] == 'Coat':
                        mask_cloth += masks[i]
            mask_face = np.clip(mask_face, 0, 1)
            mask_hair = np.clip(mask_hair, 0, 1)
            mask_neck = np.clip(mask_neck, 0, 1)
            mask_cloth = np.clip(mask_cloth, 0, 1)
            mask_human = np.clip(mask_human, 0, 1)
            if np.sum(mask_face) > 0:
                soft_mask = np.clip(mask_face, 0, 1)
                if ksize1 > 0:
                    kernel_size1 = int(np.sqrt(np.sum(soft_mask)) * ksize1)
                    kernel1 = np.ones((kernel_size1, kernel_size1))
                    soft_mask = cv2.dilate(soft_mask, kernel1, iterations=1)
                if ksize > 0:
                    kernel_size = int(np.sqrt(np.sum(soft_mask)) * ksize)
                    kernel = np.ones((kernel_size, kernel_size))
                    soft_mask_dilate = cv2.dilate(soft_mask, kernel, iterations=1)
                    if warp_mask is not None:
                        soft_mask_dilate = soft_mask_dilate * (np.clip(soft_mask + warp_mask[:, :, 0], 0, 1))
                    if eyeh > 0:
                        soft_mask = np.concatenate((soft_mask[:eyeh], soft_mask_dilate[eyeh:]), axis=0)
                    else:
                        soft_mask = soft_mask_dilate
            else:
                if ksize1 > 0:
                    kernel_size1 = int(np.sqrt(np.sum(soft_mask)) * ksize1)
                    kernel1 = np.ones((kernel_size1, kernel_size1))
                    soft_mask = cv2.dilate(mask_face, kernel1, iterations=1)
                else:
                    soft_mask = mask_face
            if include_neck:
                soft_mask = np.clip(soft_mask + mask_neck, 0, 1)

        if return_human:
            mask_human = cv2.GaussianBlur(mask_human, (21, 21), 0) * mask_human
            return soft_mask, mask_human
        else:
            return soft_mask

    def fc_segment(self, source_image):
        source_image = img_to_tensor(source_image)
        mask = self.segment(get_segmentation(), source_image, ksize=0.1)
        return (img_to_mask(mask),)
