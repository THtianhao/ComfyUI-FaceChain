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
# import pydevd_pycharm
# pydevd_pycharm.settrace('49.7.62.197', port=10090, stdoutToServer=True, stderrToServer=True)

from .model_holder import *
from .utils.img_utils import *
from .utils.convert_utils import *

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
        return (img_to_tensor(inpaint_img), mask_np3_to_mask_tensor(mask_large))

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

    def segment(self, segmentation_pipeline, img, ksize=0, eyeh=0, ksize1=0, include_neck=False, warp_mask=None, return_human=False):
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
            soft_mask = 0
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
        source_image = tensor_to_img(source_image)
        mask = self.segment(get_segmentation(), source_image, ksize=0.1)
        return (mask_np2_to_mask_tensor(mask),)

class FCReplaceImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "replace_image": ("IMAGE",),
                "face_box": ("BOX",),
                "mask": ("MASK",)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "replace_image"
    CATEGORY = "facechain/model"

    def replace_image(self, source_image, replace_image, face_box, mask):
        face_ratio = 0.45
        h, w, _ = replace_image.shape
        cropl = int(max(face_box[3] - face_box[1], face_box[2] - face_box[0]) / face_ratio / 2)
        cx = int((face_box[2] + face_box[0]) / 2)
        cy = int((face_box[1] + face_box[3]) / 2)
        cropup = min(cy, cropl)
        cropbo = min(h - cy, cropl)
        crople = min(cx, cropl)
        cropri = min(w - cx, cropl)
        ksize = int(10 * cropl / 256)
        rst_gen = cv2.resize(replace_image, (cropl * 2, cropl * 2))
        rst_crop = rst_gen[cropl - cropup:cropl + cropbo, cropl - crople:cropl + cropri]
        print(rst_crop.shape)
        inpaint_img_rst = np.zeros_like(source_image)
        print('Start pasting.')
        inpaint_img_rst[cy - cropup:cy + cropbo, cx - crople:cx + cropri] = rst_crop
        print('Fininsh pasting.')
        print(inpaint_img_rst.shape, mask.shape, source_image.shape)
        mask_large = mask.astype(np.float32)
        kernel = np.ones((ksize * 2, ksize * 2))
        mask_large1 = cv2.erode(mask_large, kernel, iterations=1)
        mask_large1 = cv2.GaussianBlur(mask_large1, (int(ksize * 1.8) * 2 + 1, int(ksize * 1.8) * 2 + 1), 0)
        mask_large1[face_box[1]:face_box[3], face_box[0]:face_box[2]] = 1
        mask_large = mask_large * mask_large1
        final_inpaint_rst = (inpaint_img_rst.astype(np.float32) * mask_large.astype(np.float32) + source_image.astype(np.float32) * (1.0 - mask_large.astype(np.float32))).astype(np.uint8)
        return (final_inpaint_rst,)

class FCCropBottom:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "face_index": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop_bottom"
    CATEGORY = "facechain/crop"

    def crop_bottom(self, source_image, width):
        source_image = tensor_to_img(source_image)
        crop_result = crop_bottom(source_image, width)
        return (img_to_tensor(crop_result),)

class FCCropFace:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "crop_ratio": ("FLOAT", {"default": 1.0, "min": 0, "max": 10, "step": 0.1})
            }
        }

    RETURN_TYPES = ("IMAGE", "BOX", "KEY_POINT")
    FUNCTION = "face_crop"
    CATEGORY = "facechain/crop"

    def face_crop(self, source_image, crop_ratio):
        source_image_pil = tensor_to_img(source_image)
        det_result = get_face_detection(source_image_pil)
        bboxes = det_result['boxes']
        keypoints = det_result['keypoints']
        area = 0
        idx = 0
        for i in range(len(bboxes)):
            bbox = bboxes[i]
            area_tmp = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area_tmp > area:
                area = area_tmp
                idx = i
        bbox = bboxes[idx]
        keypoint = keypoints[idx]
        points_array = np.zeros((5, 2))
        for k in range(5):
            points_array[k, 0] = keypoint[2 * k]
            points_array[k, 1] = keypoint[2 * k + 1]
        w, h = source_image_pil.size
        face_w = bbox[2] - bbox[0]
        face_h = bbox[3] - bbox[1]
        bbox[0] = np.clip(np.array(bbox[0], np.int32) - face_w * (crop_ratio - 1) / 2, 0, w - 1)
        bbox[1] = np.clip(np.array(bbox[1], np.int32) - face_h * (crop_ratio - 1) / 2, 0, h - 1)
        bbox[2] = np.clip(np.array(bbox[2], np.int32) + face_w * (crop_ratio - 1) / 2, 0, w - 1)
        bbox[3] = np.clip(np.array(bbox[3], np.int32) + face_h * (crop_ratio - 1) / 2, 0, h - 1)
        bbox = np.array(bbox, np.int32)
        result_image = source_image[:, bbox[3]:bbox[1], bbox[0]:bbox[2], :]
        return result_image, bbox , points_array
