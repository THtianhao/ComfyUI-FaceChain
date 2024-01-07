import numpy as np
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import insightface
from insightface.app import FaceAnalysis

image_face_fusion = None
face_recognition = None
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
        face_detection = pipeline(task=Tasks.face_detection, model='damo/cv_ddsar_face-detection_iclr23-damofd', model_revision='v1.1')
    return face_detection


def get_image_face_fusion():
    global image_face_fusion
    if image_face_fusion is None:
        image_face_fusion = pipeline('face_fusion_torch', model='damo/cv_unet_face_fusion_torch', model_revision='v1.0.5')
    return image_face_fusion


def get_segmentation():
    global segmentation
    if segmentation is None:
        segmentation = pipeline(Tasks.image_segmentation, 'damo/cv_resnet101_image-multiple-human-parsing')
    return segmentation
