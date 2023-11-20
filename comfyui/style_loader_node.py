import os
import json
import comfy.utils
import comfy
import comfy.sd
import modelscope
# from facechain.constants import neg_prompt as neg, pos_prompt_with_cloth, pos_prompt_with_style, \
#     pose_models, pose_examples, base_models, tts_speakers_map
#from facechain.utils import snapshot_download, check_ffmpeg, set_spawn_method, project_dir, join_worker_data_dir
from modelscope import snapshot_download
import folder_paths
import sys

my_dir = os.path.dirname(os.path.abspath(__file__))
custom_nodes_dir = os.path.abspath(os.path.join(my_dir, '../../ComfyUI-FaceChain'))
#comfy_dir = os.path.abspath(os.path.join(my_dir, '..', '..'))
comfy_dir = os.path.abspath(os.path.join(my_dir, '../..'))

# Append comfy_dir to sys.path & import files
sys.path.append(comfy_dir)
#all_data = []
# 名字到数据的映射
name_map = {}
style_list = []
base_models = [
    {'name': 'leosamsMoonfilm_filmGrain20',
    'model_id': 'ly261666/cv_portrait_model',
    'revision': 'v2.0',
    'sub_path': "film/film"},
    {'name': 'MajicmixRealistic_v6',
    'model_id': 'YorickHe/majicmixRealistic_v6',
    'revision': 'v1.0.0',
    'sub_path': "realistic"},
]

class FCStyleLoraLoad:
    def __init__(self):
        self.loaded_lora = None
        print("FC StyleLoraLoad initing")
        # 存储所有数据
        load_style_files()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "style_name": (load_style_files(),),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING")
    FUNCTION = "style_lora_load"
    CATEGORY = "facechain/lora"

    def style_lora_load(self, model=None, clip=None, style_name=None):
        style_data = get_data_by_name(style_name)
        base_model_name = style_data["base_model_name"]
        matching_model = next((model for model in base_models if model['name'] == base_model_name), None)
        base_model = matching_model['model_id']
        base_model_revision = matching_model['revision']
        base_model_sub_path = matching_model['sub_path']
        style_model_id = style_data["model_id"]
        style_revision = style_data["revision"]
        add_prompt_style = style_data["add_prompt_style"]
        style_bin_file = style_data["bin_file"]
        style_multiplier_style = style_data["multiplier_style"]
        style_multiplier_human = style_data["multiplier_human"]

        # matched = list(filter(lambda item: style_name == item['name'], styles))
        # if len(matched) == 0:
        #     raise ValueError(f'styles not found: {style_name}')
        # matched = matched[0]
        # style_model = matched['name']
        if style_model_id is None:
             style_model_path = None
        else:
             model_dir = snapshot_download(style_model_id, style_revision)
             style_model_path = os.path.join(model_dir, style_bin_file)
        #if using modelscope, use following code download model
        #base_model_path = snapshot_download(base_model, revision=base_model_revision)
        #if base_model_sub_path is not None and len(base_model_sub_path) > 0:
        #    base_model_path = os.path.join(base_model_path, base_model_sub_path)
        #model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
        if  matching_model['name'] == 'leosamsMoonfilm_filmGrain20':
            base_model_name = 'leosamsMoonfilm_filmGrain20.safetensors'
        if  matching_model['name'] == 'MajicmixRealistic_v6':
            base_model_name = 'MajicmixRealistic_v6.safetensors'

        ckpt_path = folder_paths.get_full_path("checkpoints", base_model_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True,
                                                    embedding_directory=folder_paths.get_folder_paths("embeddings"))

        model_patcher, clip, vae, clipvision = out

        if style_model_id is None:
            model_dir = snapshot_download('Cherrytest/zjz_mj_jiyi_small_addtxt_fromleo', revision='v1.0.0')
            style_model_path = os.path.join(model_dir, 'zjz_mj_jiyi_small_addtxt_fromleo.safetensors')
            lora = comfy.utils.load_torch_file(style_model_path, safe_load=True)
            model_lora, clip_lora = comfy.sd.load_lora_for_models(model_patcher, clip, lora, style_multiplier_style, 1)
            return (model_lora, clip_lora, vae, add_prompt_style)
        else:
            lora = comfy.utils.load_torch_file(style_model_path, safe_load=True)
            model_lora, clip_lora = comfy.sd.load_lora_for_models(model_patcher, clip, lora, style_multiplier_style, 1)
            return(model_lora, clip_lora, vae, add_prompt_style)


    def load_style_files(self):
        folder = "styles"

        for subdir in os.listdir(folder):
            subdir_path = os.path.join(folder, subdir)
            for json_file in os.listdir(subdir_path):
                json_path = os.path.join(subdir_path, json_file)

                with open(json_path, encoding='utf-8') as f:
                    json_data = json.load(f)
                #all_data.append(json_data)
                name = json_data['name']
                name_map[name] = json_data
                # style_list = json.dumps(dict(name_map.keys()))
                # print(style_list)
        return list(name_map.keys())


# 根据名称查询
# 读取文件和解析
def load_style_files():
    folder =  f"{custom_nodes_dir}/styles"
    for subdir in os.listdir(folder):
        subdir_path = os.path.join(folder, subdir)
        for json_file in os.listdir(subdir_path):
            json_path = os.path.join(subdir_path, json_file)

            with open(json_path, encoding='utf-8') as f:
                json_data = json.load(f)
                json_data["base_model_name"] = subdir

            #all_data.append(json_data)
            name = json_data['name']
            name_map[name] = json_data

            style_list = list(name_map.keys())
            #print(style_list)
    return style_list


NODE_CLASS_MAPPINGS = {
    "FCStyleLoraLoad": FCStyleLoraLoad,
}
def get_data_by_name(name):
    return name_map[name]

styles = []
# for base_model in base_models:
#     style_in_base = []
#     folder_path = f"{os.path.dirname(os.path.abspath(__file__))}/styles/{base_model['name']}"
#     files = os.listdir(folder_path)
#     files.sort()
#     for file in files:
#         file_path = os.path.join(folder_path, file)
#         with open(file_path, "r", encoding='utf-8') as f:
#             data = json.load(f)
#             # if data['img'][:2] == './':
#             #     data['img'] = f"{project_dir}/{data['img'][2:]}"
#             style_in_base.append(data['name'])
#             styles.append(data)
#     base_model['style_list'] = style_in_base


def main():
    folder = 'styles'
    # 加载文件
    fc = FCStyleLoraLoad();
    #fc.load_style_files()
    fc.style_lora_load("","",'盔甲风(Armor)')
    fc.style_lora_load("", "", "秋日胡杨风(Autumn populus euphratica style)")

    # 测试根据名称查询
    name = "秋日胡杨风(Autumn populus euphratica style)"
    print(fc.name_map[name])


if __name__ == '__main__':
    main()
