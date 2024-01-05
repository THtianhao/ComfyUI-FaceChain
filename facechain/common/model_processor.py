import cv2
import numpy as np
from facechain.model_holder import *
from facechain.utils.convert_utils import *


def debug(*args):
    print(f"==== face chain debug ====", *args)


def facechain_detect_crop(source_image_pil, face_index, crop_ratio, mode):
    det_result = get_face_detection()(source_image_pil)
    mask = np.zeros_like(source_image_pil)
    bboxes = det_result['boxes']
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
    debug('w = ', w, 'h = ', h)
    face_w = bbox[2] - bbox[0]
    face_h = bbox[3] - bbox[1]
    for k in range(5):
        points_array[k, 0] = keypoint[2 * k]
        points_array[k, 1] = keypoint[2 * k + 1]
    debug(f'mode = ', mode)
    if mode == "real seg":
        bbox[0] = np.clip(np.array(bbox[0], np.int32) - face_w * (crop_ratio - 1) / 2, 0, w - 1)
        bbox[1] = np.clip(np.array(bbox[1], np.int32) - face_h * (crop_ratio - 1) / 2, 0, h - 1)
        bbox[2] = np.clip(np.array(bbox[2], np.int32) + face_w * (crop_ratio - 1) / 2, 0, w - 1)
        bbox[3] = np.clip(np.array(bbox[3], np.int32) + face_h * (crop_ratio - 1) / 2, 0, h - 1)
        bbox = np.array(bbox, np.int32)
        mask[bbox[1]: bbox[3], bbox[0]:bbox[2]] = 1
        debug('mask', mask.shape)
        corp_img_pil = source_image_pil.crop(bbox)
        return corp_img_pil, mask, bbox, points_array
    elif mode == "square 512 width heigh":
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
