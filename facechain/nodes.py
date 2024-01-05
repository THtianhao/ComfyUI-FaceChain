# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import os

import cv2
from skimage import transform
from modelscope.outputs import OutputKeys
import pydevd_pycharm

pydevd_pycharm.settrace('49.7.62.197', port=10090, stdoutToServer=True, stderrToServer=True)

from facechain.model_holder import *
from facechain.utils.img_utils import *
from facechain.utils.convert_utils import *
from facechain.common.model_processor import facechain_detect_crop

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

class FaceDetectCrop:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "face_index": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1}),
                "crop_ratio": ("FLOAT", {"default": 1.0, "min": 0, "max": 10, "step": 0.1})
            }
        }

    RETURN_TYPES = ("IMAGE", "BOX", "KEY_POINT")
    FUNCTION = "face_detection"
    CATEGORY = "facechain/model"

    def face_detection(self, source_image, face_index, crop_ratio):
        corp_img_pil, bbox, points_array = facechain_detect_crop(tensor_to_img(source_image), face_index, crop_ratio)
        return (img_to_tensor(corp_img_pil), bbox, points_array)

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

class FCFaceSwap():
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
        pil_source_image = tensor_to_img(source_image)
        mask = self.segment(get_segmentation(), pil_source_image, ksize=0.1)
        seg_image = tensor_to_np(source_image) * mask[:, :, None]
        return (image_np_to_image_tensor(seg_image), mask_np2_to_mask_tensor(mask),)

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
