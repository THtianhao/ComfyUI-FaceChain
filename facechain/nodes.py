# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import os

import cv2
from skimage import transform
from modelscope.outputs import OutputKeys

from facechain.model_holder import *
from facechain.utils.img_utils import *
from facechain.utils.convert_utils import *
from facechain.common.model_processor import *


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
        result_image = face_fusion(source_image, fusion_image)
        return (image_np_to_image_tensor(result_image),)


class FaceDetectCrop:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "face_index": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1}),
                "crop_ratio": ("FLOAT", {"default": 1.0, "min": 0, "max": 10, "step": 0.1}),
                "mode": (["normal", "square 512 width heigh"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BOX", "KEY_POINT")
    FUNCTION = "face_detection"
    CATEGORY = "facechain/model"

    def face_detection(self, source_image, face_index, crop_ratio, mode):
        pil_image = tensor_to_img(source_image)
        corp_img_pil, mask, bbox, points_array = facechain_detect_crop(pil_image, face_index, crop_ratio, mode)
        return (image_to_tensor(corp_img_pil), mask_np3_to_mask_tensor(mask), bbox, points_array,)


class FCFaceSegment:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_image": ("IMAGE",),

            },
            "optional": {
                "ksize": ("FLOAT", {"default": 0, "min": 0, "max": 10, "step": 0.1}),
                "ksize1": ("FLOAT", {"default": 0, "min": 0, "max": 10, "step": 0.1}),
                "include_neck": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                "warp_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("seg_image", "soft_mask", "human_mask")
    FUNCTION = "fc_segment"
    CATEGORY = "facechain/model"

    def fc_segment(self, source_image, ksize=0, ksize1=0, include_neck=False, warp_mask=None, ):
        pil_source_image = tensor_to_img(source_image)
        seg_image, mask, human_mask = segment(pil_source_image, ksize, ksize1, include_neck, warp_mask, True)
        return image_to_tensor(seg_image), mask_np2_to_mask_tensor(mask), mask_np2_to_mask_tensor(human_mask)


class FCFaceFusionAndSegReplace:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "replace_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("fusion", "fusion seg replace")
    FUNCTION = "face_swap"
    CATEGORY = "facechain/model"

    def face_swap(self, source_image, replace_image):
        pil_source_image = tensor_to_img(source_image)
        pil_replace_image = tensor_to_img(replace_image)
        cv_fusion_result, cv_replace_result = face_fusing_seg_replace(pil_source_image, pil_replace_image)
        return (image_np_to_image_tensor(cv_fusion_result), image_np_to_image_tensor(cv_replace_result),)


class FCRemoveCannyFace:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "canny_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "remove_canny_face"
    CATEGORY = "facechain/model"

    def remove_canny_face(self, source_image, canny_image):
        pil_source_image = tensor_to_img(source_image)
        np_canny_image = tensor_to_np(canny_image)
        corp_img_pil, _, _, points_array = facechain_detect_crop(pil_source_image, 0, 1.1, 'normal')
        eye_height = int((points_array[0, 1] + points_array[1, 1]) / 2)
        _, mask, _ = segment(pil_source_image, ksize=0.05, eyeh=eye_height)
        canny_image = (np_canny_image * (1.0 - mask[:, :, None])).astype(np.uint8)
        return (image_np_to_image_tensor(canny_image),)


class FCCropBottom:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": 0, "max": 2048, "step": 1})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop_bottom"
    CATEGORY = "facechain/crop"

    def crop_bottom(self, source_image, width):
        source_image = tensor_to_img(source_image)
        crop_result = crop_bottom(source_image, width)
        return (image_to_tensor(crop_result),)


class FCEdgeAdd:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "edge_add_image": ("IMAGE",),
                "human_mask": ("MASK",)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "edge_add"
    CATEGORY = "facechain/crop"

    def edge_add(self, source_image, edge_add_image, human_mask):
        np_source_image = tensor_to_np(source_image)
        np_edge_add_origin_image = tensor_to_np(edge_add_image)
        np_human_mask = mask_tensor_to_mask_np3(human_mask)
        edge_add = np_source_image.astype(np.int16) - np_edge_add_origin_image.astype(np.int16)
        edge_add = edge_add * (1 - np_human_mask)
        result = Image.fromarray((np.clip(np_source_image.astype(np.int16) + edge_add.astype(np.int16), 0, 255)).astype(np.uint8))
        return (image_to_tensor(result),)


class FCReplaceByMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "replace_image": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "replace_by_mask"
    CATEGORY = "facechain/crop"

    def replace_by_mask(self, source_image, replace_image, mask):
        np_source_image = tensor_to_np(source_image)
        np_replace_image = tensor_to_np(replace_image)
        np_mask = mask_tensor_to_mask_np3(mask)
        result_np = np_source_image * np_mask + np_replace_image(1 - np_mask)
        return (image_np_to_image_tensor(result_np),)


class FCCropAndPaste:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "template_image": ("IMAGE",),
                "human_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("crop_image", "mask", "invert_mask")
    FUNCTION = "crop_and_paste"
    CATEGORY = "facechain/crop"

    def crop_and_paste(this, template_image, human_image):
        pil_template_image = tensor_to_img(template_image)
        pil_human_image = tensor_to_img(human_image)
        _, _, _, template_five_point = facechain_detect_crop(pil_template_image, 0, 1.1, 'normal')
        _, _, human_box, human_five_point = facechain_detect_crop(pil_human_image, 0, 1.5, 'normal')
        _, human_mask, _ = segment(pil_human_image)
        human_mask = np.expand_dims((human_mask * 255).astype(np.uint8), axis=2)
        human_mask = np.concatenate([human_mask, human_mask, human_mask], axis=2)
        pil_human_mask = Image.fromarray(human_mask)
        output, mask = crop_and_paste(pil_human_image, pil_human_mask, pil_template_image, human_five_point, template_five_point, human_box)
        return image_np_to_image_tensor(output), mask_np3_to_mask_tensor(1 - mask), mask_np3_to_mask_tensor(mask)


class FCMaskOP:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "method": (["expand_dims", "concatenate"],),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "mask_op"
    CATEGORY = "facechain/mask"

    def mask_op(self, mask, method):
        mask = mask_tensor_to_mask_np3(mask)
        result = None
        if method == "concatenate":
            result = np.concatenate([mask, mask, mask], axis=2)
        elif method == "expand_dims":
            result = np.expand_dims(mask, axis=2)
        return (mask_np3_to_mask_tensor(result),)
