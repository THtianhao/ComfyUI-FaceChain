# [ComfyUI-FaceChain](https://github.com/THtianhao/ComfyUI-FaceChain)

This project is adapted from [facechain](https://github.com/modelscope/facechain) and involves the breakdown and improvement of the processes of [facechain](https://github.com/modelscope/facechain).

![workflow_inpaiting_inference.png](workflows%2Fworkflow_inpaiting_inference.png)

English | [简体中文](./README_zh-CN.md)

## Contact

If you have any questions or suggestions, you can reach us through the following channels:

- Email: tototianhao@gmail.com
- Telegram: https://t.me/+JoFE2vqHU4phZjg1
- WeChat Group: <img src="./images/wechat.jpg" width="300">

### Steps
1. Install ComfyUI first.

2. After a successful installation of ComfyUI, navigate to the `custom_nodes` directory at `ComfyUI/custom_nodes/`.

```
cd custom_nodes
```

3. Clone this project into the `custom_nodes` directory.

```
git clone https://github.com/THtianhao/ComfyUI-FaceChain
```

4. Restart ComfyUI.

## ComfyUI Workflow

Facechain Workflow Location: [workflow_inpaiting_inference.png](workflows%2Fworkflow_inpaiting_inference.png)
Drag the workflow directly into ComfyUI.

## Node Introduction
### FC StyleLoraLoad
  > The workflow can load the checkpoints and style Lora used by facechain, download them first, and then merge them, providing relevant prompts.
  > Workflow: [./workflow/workflow_inference.json](./workflows/workflow_inference.json)
### FC FaceDetectCrop
  > Detects faces and crops them.
   
 ![workflow_face_detect_crop.png](workflows%2Fworkflow_face_detect_crop.png) 
  Parameter Description:

  1. mode: Cropping mode, normal mode crops according to the face, square 512 width height will scale the face to 512.
  2. face_index: Index of the face, if there are multiple faces, retrieve them based on the index.
  3. crop_ratio: Only effective in normal mode, crops the face proportionally, 1.0 is 1x the face.

### FC FaceFusion
  > Fusion using model scope model.
 
 ![workflow_face_fusion.png](workflows%2Fworkflow_face_fusion.png) 

  
### FC FaceSegment
  > Segmentation using model scope model to obtain masks for the face and body.

![workflow_face_segment.png](workflows%2Fworkflow_face_segment.png)  
  
Parameter Description:
1. ksize: Expansion parameter for segmenting the edges of the face.
2. ksize1: Expansion parameter for segmenting the edges of the face.
3. include_neck: Whether the segmented image includes the neck.

### FC FaceSegAndReplace
  > Performs face fusion and replaces the original image, similar to facefusion but mainly used for multiple people.

![workflow_face_seg_and_replace.png](workflows%2Fworkflow_face_seg_and_replace.png)  


### FC RemoveCannyFace
  > Removes the Canny-detected parts of the face.
  
![workflow_remove_canny_face.png](workflows%2Fworkflow_remove_canny_face.png)
### FC ReplaceByMask
  > Replaces the image based on the mask.
  
![workflow_replace_by_mask.png](workflows%2Fworkflow_replace_by_mask.png)

* FC MaskOP
  > Operations on the mask.

Parameter Description:
1. mode: Provides three operations, blur, erosion, dilation.
2. kernel: The kernel used for the operation, the larger the kernel, the stronger the operation.
  
![workflow_mask_op.png](workflows%2Fworkflow_mask_op.png)

* FC FCCropToOrigin
  > Currently, it can only be used in conjunction with `FC FaceDetectCrop` in `square 512 width height` mode. Pastes the cropped image onto the target image based on the mask.
  
![workflow_crop_to_origin.png](workflows%2Fworkflow_crop_to_origin.png)
Parameter Description:
1. origin_image: Original image.
2. origin_box: Bounding box of the original image.
3. origin_mask: Mask cropped from the original image.
4. paste_image: Pasting image, must be consistent with the origin_mask, hence the need for `FC FaceDetectCrop` in `square 512 width height` mode.
  



## Contribution

If you find any issues or have suggestions for improvement, feel free to contribute. Follow these steps:

1. Branch out a new feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes and commit: `git commit -m "Add new feature"`
3. Push to your remote branch: `git push origin feature/your-feature-name`
4. Create a Pull Request (PR).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.



Join us and contribute to the development of EasyPhoto ComfyUI Plugin!
