# [ComfyUI-FaceChain](https://github.com/THtianhao/ComfyUI-FaceChain)

项目改编于[facechain](https://github.com/modelscope/facechain)，对于[facechain](https://github.com/modelscope/facechain)进行了流程上的拆解和改进。

![workflow_inpaiting_inference.png](workflows%2Fworkflow_inpaiting_inference.png)

English | [简体中文](./README_zh-CN.md)

## 联系

如果你有任何疑问或建议，可以通过以下方式联系我们：

- 电子邮件：tototianhao@gmail.com
- telegram: https://t.me/+JoFE2vqHU4phZjg1
- 微信群： <img src="./images/wechat.jpg" width="300">

### 步骤
1. 首先安装ComfyUI

2. ComfyUI运行成功后进入`custom_nodes` 目录 `ComfyUI/custom_nodes/`

```
cd custom_nodes
```

3. 克隆此项目到custom_nodes目录中

```
git clone https://github.com/THtianhao/ComfyUI-FaceChain
```

4. 重新启动ComfyUI

## ComfyUI 工作流

Facechain工作位置: [workflow_inpaiting_inference.png](workflows%2Fworkflow_inpaiting_inference.png)
将工作流直接拖进comfyui中

## 节点介绍
### FC StyleLoraLoad
  > workflow可以加载facechain 用到的checkpoint和风格lora，首先下载然后进行融合，并且给出相关的prompt
  > workflow ： [./workflow/workflow_inference.json](./workflows/workflow_inference.json)
### FC FaceDetectCrop
  > 识别人脸并且剪裁
   
 ![workflow_face_detect_crop.png](workflows%2Fworkflow_face_detect_crop.png) 
  参数介绍：

  1. mode:剪裁模式，normal模式为按照人脸剪裁,square 512 width height会将人脸缩放到512
  2. face_index:人脸的索引，如果有多个人脸的话按照多个人脸的话按照index进行获取
  3. crop_ratio:只在normal模式下生效，将人脸按照比例剪裁1.0为1倍人脸

### FC FaceFusion
  > 使用model scope模型进行融合
 
 ![workflow_face_fusion.png](workflows%2Fworkflow_face_fusion.png) 

  
### FC FaceSegment
  > 使用model scope模型进行分割并且获取脸部和身体的mask

![workflow_face_segment.png](workflows%2Fworkflow_face_segment.png)  
  
参数介绍:
1. ksize: 分割人脸边缘的扩展参数
2. ksize1: 分割人脸边缘的扩展参数
3. include_neck: 分割的图像是否包含脖子

### FC FaceSegAndReplace
  > 进行人脸融合并且分割人脸替换原图,差异和facefusion不大,主要用在多人

![workflow_face_seg_and_replace.png](workflows%2Fworkflow_face_seg_and_replace.png)  


### FC RemoveCannyFace
  > 删除掉canny的人脸部分
  
![workflow_remove_canny_face.png](workflows%2Fworkflow_remove_canny_face.png)
### FC ReplaceByMask
  > 根据mask替换图像
  
![workflow_replace_by_mask.png](workflows%2Fworkflow_replace_by_mask.png)

* FC MaskOP
  > 对mask的操作

参数介绍:
1. mode: 提供了三种操作，模糊处理，腐蚀，膨胀
2. kernel: 用于操作的核，越大操作力度越强
  
![workflow_mask_op.png](workflows%2Fworkflow_mask_op.png)

* FC FCCropToOrigin
  > 目前只能配合 `FC FaceDetectCrop` 的`square 512 width height`模式一起使用,将截取的图像根据mask粘贴到目标图像上
  
![workflow_crop_to_origin.png](workflows%2Fworkflow_crop_to_origin.png)![workflow_face_detect_crop.png](workflows%2Fworkflow_face_detect_crop.png)

参数介绍:
1. origin_image: 原始图像
2. origin_box:原始图像的bbox
3. origin_mask:原始图像截取的mask
4. paste_image:粘贴图像 必须和origin_mask保持一致,因此需要`FC FaceDetectCrop` 的`square 512 width height`模式
  



## 贡献

如果你发现任何问题或有改进建议，欢迎贡献。请遵循以下步骤：

1. 分支出一个新的特性分支：`git checkout -b feature/your-feature-name`
2. 进行修改并提交：`git commit -m "Add new feature"`
3. 推送到你的远程分支：`git push origin feature/your-feature-name`
4. 创建一个 Pull 请求（PR）。

## 许可证

该项目采用 MIT 许可证。查看 [LICENSE](LICENSE) 文件以获取详细信息。



欢迎加入我们，为 EasyPhoto ConfyUI Plugin 的发展做出贡献！
