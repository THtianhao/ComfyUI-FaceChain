import os
import json

import comfy.utils
import comfy
import comfy.sd
from modelscope import snapshot_download
import folder_paths
import sys

# import pydevd_pycharm
# pydevd_pycharm.settrace('49.7.62.197', port=10090, stdoutToServer=True, stderrToServer=True, suspend=False)

my_dir = os.path.dirname(os.path.abspath(__file__))
custom_nodes_dir = os.path.abspath(os.path.join(my_dir, '../../ComfyUI-FaceChain'))
comfy_dir = os.path.abspath(os.path.join(my_dir, '../..'))

sys.path.append(comfy_dir)

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
    name_map = {}
    style_key_set = None

    @classmethod
    def INPUT_TYPES(s):
        folder = f"{custom_nodes_dir}/styles"
        for subdir in os.listdir(folder):
            subdir_path = os.path.join(folder, subdir)
            for json_file in os.listdir(subdir_path):
                json_path = os.path.join(subdir_path, json_file)

                with open(json_path, encoding='utf-8') as f:
                    json_data = json.load(f)
                    json_data["base_model_name"] = subdir

                # all_data.append(json_data)
                name = json_data['name']
                s.name_map[name] = json_data
        s.style_key_set = list(s.name_map.keys())
        return {
            "required": {
                "style_name": (s.style_key_set,)
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING",)
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "style_prompt",)
    FUNCTION = "style_lora_load"
    CATEGORY = "facechain/lora"

    def style_lora_load(self, style_name=None):
        style_data = self.name_map[style_name]
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
        # if using modelscope, use following code download model
        if matching_model['name'] == 'leosamsMoonfilm_filmGrain20':
            base_model_name = 'leosamsMoonfilm_filmGrain20.safetensors'
        if matching_model['name'] == 'MajicmixRealistic_v6':
            base_model_name = 'majicmixRealistic_v6.safetensors'

        ckpt_path = folder_paths.get_full_path("checkpoints", base_model_name)
        # check if modelfile not exist, download it from modelscope
        if ckpt_path == None:
            model_cache_dir = snapshot_download(f'ultimatech/{matching_model["name"]}')
            # shutil.move(os.path.join(model_cache_dir,base_model_name),folder_paths.folder_names_and_paths["checkpoints"][0][0])
            # shutil.rmtree(model_cache_dir)
            ckpt_path = os.path.join(model_cache_dir, base_model_name)
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
            return (model_lora, clip_lora, vae, add_prompt_style)

def load_style_files():
    name_map = {}
    folder = f"{custom_nodes_dir}/styles"
    for subdir in os.listdir(folder):
        subdir_path = os.path.join(folder, subdir)
        for json_file in os.listdir(subdir_path):
            json_path = os.path.join(subdir_path, json_file)

            with open(json_path, encoding='utf-8') as f:
                json_data = json.load(f)
                json_data["base_model_name"] = subdir

            # all_data.append(json_data)
            name = json_data['name']
            name_map[name] = json_data
    key_set = list(name_map.keys())
    print(f'style name list ======= ', key_set)

def main():
    load_style_files()
    folder = 'styles'
    # 加载文件
    fc = FCStyleLoraLoad();
    # fc.load_style_files()
    fc.style_lora_load("", "", '盔甲风(Armor)')
    fc.style_lora_load("", "", "秋日胡杨风(Autumn populus euphratica style)")

    # 测试根据名称查询
    name = "秋日胡杨风(Autumn populus euphratica style)"
    print(fc.name_map[name])

if __name__ == '__main__':
    main()
