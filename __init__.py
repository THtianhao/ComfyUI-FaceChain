import os
import subprocess
import sys
import threading

root_path = os.path.dirname(__file__)
parent_dir = os.path.dirname(root_path)
sys.path.append(root_path)
from facechain.nodes import *
from facechain.style_loader_node import *

NODE_CLASS_MAPPINGS = {
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


# install_model = ["python-slugify==8.0.1", "modelscope", "controlnet_aux==0.0.6", "onnxruntime==1.15.1", "mmcv==1.7.0", "mmdet==2.26.0", "mediapipe==0.10.3", "edge_tts"]

def handle_stream(stream, prefix):
    for line in stream:
        print(prefix, line, end="")


def run_script(cmd, cwd='.'):
    process = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    stdout_thread = threading.Thread(target=handle_stream, args=(process.stdout, ""))
    stderr_thread = threading.Thread(target=handle_stream, args=(process.stderr, "[!]"))
    stdout_thread.start()
    stderr_thread.start()
    stdout_thread.join()
    stderr_thread.join()
    return process.wait()


if os.path.basename(parent_dir) == "custom_nodes":
    print("##  installing facechain dependencies")
    requirements_path = os.path.join(root_path, "requirements.txt")
    run_script([sys.executable, '-s', '-m', 'pip', 'install', '-q', '-r', requirements_path])
    # for model in install_model:
    #     run_script([sys.executable, '-s', '-m', 'pip', 'install', '-q', model])
