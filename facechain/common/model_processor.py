import numpy as np
from facechain.model_holder import *

def facechain_detect_crop(source_image_pil, face_index, crop_ratio):
    det_result = get_face_detection()(source_image_pil)
    bboxes = det_result['boxes']
    keypoints = det_result['keypoints']
    area = 0
    # for i in range(len(bboxes)):
    #     bbox = bboxes[i]
    #     area_tmp = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    #     if area_tmp > area:
    #         area = area_tmp
    #         idx = i
    bbox = bboxes[face_index]
    keypoint = keypoints[face_index]
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
    corp_img_pil = source_image_pil.crop(bbox)
    return corp_img_pil, bbox, points_array
