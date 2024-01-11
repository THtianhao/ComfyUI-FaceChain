import os
import subprocess
import sys
import threading
root_path = os.path.dirname(__file__)
parent_dir = os.path.dirname(root_path)
sys.path.append(root_path)
import facechain.utils.install
from facechain.nodes import *
from facechain.style_loader_node import *
WEB_DIRECTORY = "js"


NODE_CLASS_MAPPINGS = {
    "FC StyleLoraLoad": FCStyleLoraLoad,
    "FC FaceDetectCrop": FaceDetectCrop,
    "FC FaceFusion": FCFaceFusion,
    "FC FaceSegment": FCFaceSegment,
    "FC FaceSegAndReplace": FCFaceFusionAndSegReplace,
    "FC RemoveCannyFace": FCRemoveCannyFace,
    "FC CropBottom": FCCropBottom,
    "FC ReplaceByMask": FCReplaceByMask,
    "FC CropAndPaste": FCCropAndPaste,
    "FC MaskOP": FCMaskOP,
    "FC CropToOrigin": FCCropToOrigin,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FC StyleLoraLoad": "FC StyleLoraLoad",
    "FC FaceDetectCrop": "FC FaceDetectCrop",
    "FC FaceFusion": "FC FaceFusion",
    "FC FaceSegment": "FC FaceSegment",
    "FC FaceSegAndReplace": "FC FaceSegAndReplace",
    "FC RemoveCannyFace": "FC RemoveCannyFace",
    "FC CropBottom": "FC CropBottom",
    "FC ReplaceByMask": "FC ReplaceByMask",
    "FC CropAndPaste": "FC CropAndPaste",
    "FC MaskOP": "FC MaskOP",
    "FC CropToOrigin": "FC CropToOrigin",

}
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']



