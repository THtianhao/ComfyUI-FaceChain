import numpy as np
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import insightface
from insightface.app import FaceAnalysis

image_face_fusion = None
face_recognition =  None
face_detection = None
segmentation = None

def get_face_recognition():
    global face_recognition
    if face_recognition is None:
        face_recognition = pipeline(Tasks.face_recognition, 'damo/cv_ir_face-recognition-ood_rts', model_revision='v2.5')
    return face_recognition

def get_face_detection():
    global face_detection
    if face_detection is None:
        face_detection= pipeline(task=Tasks.face_detection, model='damo/cv_ddsar_face-detection_iclr23-damofd', model_revision='v1.1')
    return face_detection

def call_face_crop(det_pipeline, image, crop_ratio):
    det_result = det_pipeline(image)
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
    w, h = image.size
    face_w = bbox[2] - bbox[0]
    face_h = bbox[3] - bbox[1]
    bbox[0] = np.clip(np.array(bbox[0], np.int32) - face_w * (crop_ratio - 1) / 2, 0, w - 1)
    bbox[1] = np.clip(np.array(bbox[1], np.int32) - face_h * (crop_ratio - 1) / 2, 0, h - 1)
    bbox[2] = np.clip(np.array(bbox[2], np.int32) + face_w * (crop_ratio - 1) / 2, 0, w - 1)
    bbox[3] = np.clip(np.array(bbox[3], np.int32) + face_h * (crop_ratio - 1) / 2, 0, h - 1)
    bbox = np.array(bbox, np.int32)
    return bbox, points_array

def get_image_face_fusion():
    global image_face_fusion
    if image_face_fusion is None:
        image_face_fusion = pipeline(Tasks.image_face_fusion, model='damo/cv_unet-image-face-fusion_damo', model_revision='v1.3')
    return image_face_fusion

def get_segmentation():
    global segmentation
    if segmentation is None:
        segmentation = pipeline(Tasks.image_segmentation, 'damo/cv_resnet101_image-multiple-human-parsing')
    return segmentation