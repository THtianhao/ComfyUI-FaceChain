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
        result_image = get_image_face_fusion()(dict(template=source_image, user=fusion_image))[OutputKeys.OUTPUT_IMG]
        result_image = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        return (image_to_tensor(result_image),)


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
                "mode": (["real seg", "square 512 width heigh"],),
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
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    FUNCTION = "fc_segment"
    CATEGORY = "facechain/model"

    def fc_segment(self, source_image):
        pil_source_image = tensor_to_img(source_image)
        mask = segment(pil_source_image, ksize=0.1)
        seg_image = tensor_to_np(source_image) * mask[:, :, None]
        return (image_np_to_image_tensor(seg_image), mask_np2_to_mask_tensor(mask),)


class FCFaceSegAndReplace:
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
    FUNCTION = "face_swap"
    CATEGORY = "facechain/model"

    def face_swap(self, source_image, replace_image):
        pil_source_image = image_to_tensor(source_image)
        pil_replace_image = image_to_tensor(replace_image)
        image = face_fusing_seg_replace(pil_source_image, pil_replace_image)
        return (image_np_to_image_tensor(image),)


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
        return (image_to_tensor(crop_result),)


class FCCropAndPaste:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "source_image_mask": ("MASK",),
                "source_box": ("BOX",),
                "source_five_point": ("KEY_POINT",),
                "target_image": ("IMAGE",),
                "target_five_point": ("KEY_POINT",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "crop_and_paste"
    CATEGORY = "facechain/crop"

    def crop_and_paste(this, source_image, source_image_mask, source_box, source_five_point, target_image, target_five_point, use_warp=True):
        source_image = tensor_to_img(source_image)
        target_image = tensor_to_img(target_image)
        source_image_mask = tensor_to_img(source_image_mask)
        if use_warp:
            source_five_point = np.reshape(source_five_point, [5, 2]) - np.array(source_box[:2])
            target_five_point = np.reshape(target_five_point, [5, 2])

            Crop_Source_image = source_image.crop(np.int32(source_box))
            Crop_Source_image_mask = source_image_mask.crop(np.int32(source_box))
            source_five_point, target_five_point = np.array(source_five_point), np.array(target_five_point)

            tform = transform.SimilarityTransform()
            tform.estimate(source_five_point, target_five_point)
            M = tform.params[0:2, :]

            warped = cv2.warpAffine(np.array(Crop_Source_image), M, np.shape(target_image)[:2][::-1], borderValue=0.0)
            warped_mask = cv2.warpAffine(np.array(Crop_Source_image_mask), M, np.shape(target_image)[:2][::-1], borderValue=0.0)

            mask = np.float32(warped_mask == 0)
            output = mask * np.float32(target_image) + (1 - mask) * np.float32(warped)
        else:
            mask = np.float32(np.array(source_image_mask) == 0)
            output = mask * np.float32(target_image) + (1 - mask) * np.float32(source_image)
        return image_np_to_image_tensor(output), mask_np3_to_mask_tensor(mask)


class FCMaskOP:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "method": (["concatenate"],),
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
        return (mask_np3_to_mask_tensor(result),)
