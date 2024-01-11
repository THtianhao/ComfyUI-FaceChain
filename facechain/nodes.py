from facechain.utils.img_utils import *
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
                "mode": (["normal", "square 512 width height"],),
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
                "ksize": ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.01}),
                "ksize1": ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.01}),
                "include_neck": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                "warp_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("seg_image", "soft_mask", "human_mask")
    FUNCTION = "fc_segment"
    CATEGORY = "facechain/model"

    def fc_segment(self, source_image, ksize, ksize1, include_neck=False, warp_mask=None, ):
        pil_source_image = tensor_to_img(source_image)
        seg_image, mask, human_mask = segment(pil_source_image, ksize=ksize, ksize1=ksize1, include_neck=include_neck, warp_mask=warp_mask, return_human=True)
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
        result_np = np_source_image * np_mask + np_replace_image * (1 - np_mask)
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
                "method": (["burl", "erode", "dilate"],),
                "kernel": ("INT", {"default": 16, "min": 0, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "mask_op"
    CATEGORY = "facechain/mask"

    def mask_op(self, mask, method, kernel):
        mask_np = mask_tensor_to_mask_np3(mask)
        result = None
        kernel_real = np.ones((kernel, kernel), np.uint8)
        if method == "burl":
            result = cv2.GaussianBlur(mask_np, (int(kernel * 1.8) * 2 + 1, int(kernel * 1.8) * 2 + 1), 0)
            result = np.expand_dims(result, axis=2)
        elif method == 'erode':
            result = cv2.erode(mask_np, kernel_real, iterations=1)
            result = np.expand_dims(result, axis=2)
        elif method == 'dilate':
            result = cv2.dilate(mask_np, kernel_real, iterations=1)
            result = np.expand_dims(result, axis=2)
        return (mask_np3_to_mask_tensor(result),)

class FCCropToOrigin:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "origin_image": ("IMAGE",),
                "origin_box": ("BOX",),
                "origin_mask": ("MASK",),
                "paste_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "past_to_origin"
    CATEGORY = "facechain/mask"

    def past_to_origin(self, origin_image, origin_box, origin_mask, paste_image):
        np_origin_image = tensor_to_np(origin_image)
        np_origin_mask = mask_tensor_to_mask_np3(origin_mask)
        np_paste_image = tensor_to_np(paste_image)

        h, w = np_origin_image.shape[:2]
        face_w = origin_box[2] - origin_box[0]
        face_h = origin_box[3] - origin_box[1]
        face_ratio = 0.45
        cropl = int(max(face_h, face_w) / face_ratio / 2)
        cx = int((origin_box[2] + origin_box[0]) / 2)
        cy = int((origin_box[1] + origin_box[3]) / 2)
        cropup = min(cy, cropl)
        cropbo = min(h - cy, cropl)
        crople = min(cx, cropl)
        cropri = min(w - cx, cropl)

        # 恢复到原来扩展到512之前的的大小
        resize_paste_image = cv2.resize(np_paste_image, (cropl * 2, cropl * 2))
        # 剪裁恢复
        resize_past_crop = resize_paste_image[cropl - cropup:cropl + cropbo, cropl - crople:cropl + cropri]

        origin_img_black = np.zeros_like(np_origin_image)
        # 根据之前的大小来贴到mask上面
        origin_img_black[cy - cropup:cy + cropbo, cx - crople:cx + cropri] = resize_past_crop
        ksize = int(10 * cropl / 256)
        kernel = np.ones((ksize * 2, ksize * 2))
        # 腐蚀操作 让区域小一点
        np_origin_mask = cv2.erode(np_origin_mask, kernel, iterations=1)
        # 模糊处理
        # np_origin_mask = cv2.GaussianBlur(np_origin_mask, (int(ksize * 1.8) * 2 + 1, int(ksize * 1.8) * 2 + 1), 0)
        # 扩展维度
        np_origin_mask = np.expand_dims(np_origin_mask, axis=2)
        # 识别的人脸区域设置为空
        np_origin_mask[origin_box[1]:origin_box[3], origin_box[0]:origin_box[2]] = 1

        result = (origin_img_black.astype(np.float32) * np_origin_mask.astype(np.float32) + np_origin_image.astype(np.float32) * (
                1.0 - np_origin_mask.astype(np.float32))).astype(np.uint8)
        return (image_np_to_image_tensor(result),)
