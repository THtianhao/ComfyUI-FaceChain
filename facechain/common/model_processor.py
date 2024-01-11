import cv2
import numpy as np
from modelscope.outputs import OutputKeys
from skimage import transform

from facechain.model_holder import *
from facechain.utils.convert_utils import *

def debug(*args):
    print(f"==== face chain debug ====", *args)

def facechain_detect_crop(source_image_pil, face_index=0, crop_ratio=1, mode='normal'):
    det_result = get_face_detection()(source_image_pil)
    mask = np.zeros_like(source_image_pil)
    bboxes = det_result['boxes']
    bboxes = np.array(bboxes).astype(np.int16)
    keypoints = det_result['keypoints']
    area = 0
    # for i in range(len(bboxes)):
    #     bbox = bboxes[i]
    #     area_tmp = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    #     if area_tmp > area:
    #         area = area_tmp
    #         idx = i
    try:
        bbox = bboxes[face_index]
    except:
        raise RuntimeError('No face detected or face index error/没有检测到人脸或者人脸的索引错误')

    keypoint = keypoints[face_index]
    points_array = np.zeros((5, 2))
    w, h = source_image_pil.size
    debug(f'bbox', bbox[0], bbox[1], bbox[2], bbox[3])
    debug('w = ', w, 'h = ', h)
    face_w = bbox[2] - bbox[0]
    face_h = bbox[3] - bbox[1]
    for k in range(5):
        points_array[k, 0] = keypoint[2 * k]
        points_array[k, 1] = keypoint[2 * k + 1]
    debug(f'mode = ', mode)
    if mode == "normal":
        bbox[0] = np.clip(np.array(bbox[0], np.int32) - face_w * (crop_ratio - 1) / 2, 0, w - 1)
        bbox[1] = np.clip(np.array(bbox[1], np.int32) - face_h * (crop_ratio - 1) / 2, 0, h - 1)
        bbox[2] = np.clip(np.array(bbox[2], np.int32) + face_w * (crop_ratio - 1) / 2, 0, w - 1)
        bbox[3] = np.clip(np.array(bbox[3], np.int32) + face_h * (crop_ratio - 1) / 2, 0, h - 1)
        bbox = np.array(bbox, np.int32)
        mask[bbox[1]: bbox[3], bbox[0]:bbox[2]] = 1
        debug('mask', mask.shape)
        corp_img_pil = source_image_pil.crop(bbox)
        return corp_img_pil, mask, bbox, points_array
    elif mode == "square 512 width height":
        np_image = image_to_np(source_image_pil)
        face_ratio = 0.45
        crop_l = int(max(face_h, face_w) / face_ratio / 2)
        cx = int((bbox[2] + bbox[0]) / 2)
        cy = int((bbox[1] + bbox[3]) / 2)
        crop_up = min(cy, crop_l)
        crop_bo = min(h - cy, crop_l)
        crop_le = min(cx, crop_l)
        crop_ri = min(w - cx, crop_l)
        debug(crop_l, cx, cy, crop_up, crop_bo, crop_le, crop_ri)
        inpaint_img = np.pad(np_image[cy - crop_up:cy + crop_bo, cx - crop_le:cx + crop_ri],
                             ((crop_l - crop_up, crop_l - crop_bo), (crop_l - crop_le, crop_l - crop_ri), (0, 0)), 'constant')
        inpaint_img = cv2.resize(inpaint_img, (512, 512))
        inpaint_img = Image.fromarray(cv2.cvtColor(inpaint_img[:, :, ::-1], cv2.COLOR_BGR2RGB))
        mask[cy - crop_up:cy + crop_bo, cx - crop_le:cx + crop_ri] = 1
        return inpaint_img, mask, bbox, points_array,
    else:
        raise RuntimeError('模式错误')

def segment(img, ksize=0, eyeh=0, ksize1=0, include_neck=False, warp_mask=None, return_human=True):
    print('ksize', ksize, "ksize1", ksize1, 'warp_mask', warp_mask)
    seg_image = get_segmentation()(img)
    masks = seg_image['masks']
    scores = seg_image['scores']
    labels = seg_image['labels']
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
    np_image = image_to_np(img)
    img.crop()
    seg_image_np = np_image * soft_mask[:, :, None]
    seg_image = Image.fromarray(seg_image_np.astype(np.uint8))
    if return_human:
        mask_human = cv2.GaussianBlur(mask_human, (21, 21), 0) * mask_human
        return seg_image, soft_mask, mask_human
    else:
        # 返回一个是PIL，一个是np的二维数组
        return seg_image, soft_mask,

def face_fusion(image, fusion_image):
    cv_fusion_result = get_image_face_fusion()(dict(template=image, user=fusion_image))[OutputKeys.OUTPUT_IMG]
    np_fusion_result = cv2.cvtColor(cv_fusion_result, cv2.COLOR_BGR2RGB)
    debug("cv_fusion_result", cv_fusion_result.shape)
    return np_fusion_result

def face_fusing_seg_replace(image, replace_image):
    np_fusion_result = face_fusion(image, replace_image)
    _, face_mask, _ = segment(image, ksize=0.1)
    cv_replace_result = (np_fusion_result * face_mask[:, :, None] + np.array(image) * (1 - face_mask[:, :, None])).astype(np.uint8)
    debug("cv_replace_result", cv_replace_result.shape)
    return np_fusion_result, cv_replace_result

def crop_and_paste(Source_image, Source_image_mask, Target_image, Source_Five_Point, Target_Five_Point, Source_box, use_warp=True):
    debug(f"crop and paste", Source_image, Source_image_mask, Target_image, Source_Five_Point, Target_Five_Point, Source_box)
    if use_warp:
        Source_Five_Point = np.reshape(Source_Five_Point, [5, 2]) - np.array(Source_box[:2])
        Target_Five_Point = np.reshape(Target_Five_Point, [5, 2])

        Crop_Source_image = Source_image.crop(np.int32(Source_box))
        Crop_Source_image_mask = Source_image_mask.crop(np.int32(Source_box))
        Source_Five_Point, Target_Five_Point = np.array(Source_Five_Point), np.array(Target_Five_Point)

        tform = transform.SimilarityTransform()
        tform.estimate(Source_Five_Point, Target_Five_Point)
        M = tform.params[0:2, :]

        warped = cv2.warpAffine(np.array(Crop_Source_image), M, np.shape(Target_image)[:2][::-1], borderValue=0.0)
        warped_mask = cv2.warpAffine(np.array(Crop_Source_image_mask), M, np.shape(Target_image)[:2][::-1], borderValue=0.0)

        mask = np.float32(warped_mask == 0)
        debug('target shape', np.float32(Target_image).shape)
        debug('warped shape', np.float32(warped).shape)
        output = mask * np.float32(Target_image) + (1 - mask) * np.float32(warped)
    else:
        mask = np.float32(np.array(Source_image_mask) == 0)
        output = mask * np.float32(Target_image) + (1 - mask) * np.float32(Source_image)
    return output, mask
