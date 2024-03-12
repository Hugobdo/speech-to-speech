import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
from PIL import Image
import json
import threading
import queue
from reasoning import MainChatModel

class PersonaImageCreator:
    def __init__(self, persona, reasoning=MainChatModel(temperature=0, max_tokens=-1)):
        self.path = r"C:\Users\T-GAMER\Desktop\Projetos\ComfyUI_windows_portable\ComfyUI"
        self.reasoning = reasoning
        self.persona = persona

        with open(f"personas/{self.persona}.json", "r") as f:
            data = json.load(f)
            self.original_appearance = data['appearance']
            self.appearance = data['appearance']
            self.model = data.get('model', 'hassakuHentaiModel_v13.safetensors')

    def create_image_prompt(self):      
        prompt = ""
        for key, value in self.appearance.items():
            if value == "":
                value = f"No {key}"
            prompt += f"{value}, "
        return prompt

    def change_appearance(self, prompt):
        self.appearance = self.reasoning.change_sd_prompt(self.appearance, prompt)

    def get_value_at_index(self, obj: Union[Sequence, Mapping], index: int) -> Any:
        try:
            return obj[index]
        except KeyError:
            return obj["result"][index]

    def find_path(self, name: str, path: str = None) -> str:
        if path is None:
            path = os.getcwd()
        if name in os.listdir(path):
            path_name = os.path.join(path, name)
            print(f"{name} found: {path_name}")
            return path_name
        parent_directory = os.path.dirname(path)
        if parent_directory == path:
            return None
        return self.find_path(name, parent_directory)

    def add_comfyui_directory_to_sys_path(self) -> None:
        comfyui_path = self.find_path("ComfyUI", path=self.path)
        if comfyui_path is not None and os.path.isdir(comfyui_path):
            sys.path.append(comfyui_path)
            print(f"'{comfyui_path}' added to sys.path")

    def add_extra_model_paths(self) -> None:
        from main import load_extra_path_config

        extra_model_paths = self.find_path("extra_model_paths.yaml", path=self.path)
        if extra_model_paths is not None:
            load_extra_path_config(extra_model_paths)
        else:
            print("Could not find the extra_model_paths config file.")

    def run(self, prompt, batch_size=1):
        self.add_comfyui_directory_to_sys_path()
        self.add_extra_model_paths()
        # Importações que dependem de caminhos dinâmicos adicionados ao sys.path
        from nodes import (
            EmptyLatentImage,
            LoraLoader,
            VAEDecode,
            NODE_CLASS_MAPPINGS,
            SaveImage,
            CLIPTextEncode,
            CheckpointLoaderSimple,
            VAELoader,
            KSampler,
        )

        with torch.inference_mode():
            checkpointloadersimple = CheckpointLoaderSimple()
            checkpointloadersimple_4 = checkpointloadersimple.load_checkpoint(
                ckpt_name=self.model
            )

            emptylatentimage = EmptyLatentImage()
            emptylatentimage_5 = emptylatentimage.generate(
                width=512, height=512, batch_size=1
            )

            cliptextencode = CLIPTextEncode()
            cliptextencode_6 = cliptextencode.encode(
                text=prompt,
                clip=self.get_value_at_index(checkpointloadersimple_4, 1),
            )

            cliptextencode_7 = cliptextencode.encode(
                text="(worst quality, low quality:1.4), malformed fingers, malformed hands, loli, malformed smile, missing fingers, missing limbs, malformed nose, eye anomaly, mouth anomaly, two heads",
                clip=self.get_value_at_index(checkpointloadersimple_4, 1)
            )

            vaeloader = VAELoader()
            vaeloader_14 = vaeloader.load_vae(vae_name="clearvae_v23.safetensors")

            ksampler = KSampler()
            vaedecode = VAEDecode()
            saveimage = SaveImage()

            for q in range(batch_size):
                ksampler_3 = ksampler.sample(
                    seed=random.randint(1, 2**64),
                    steps=21,
                    cfg=8,
                    sampler_name="euler_ancestral",
                    scheduler="exponential",
                    denoise=1,
                    model=self.get_value_at_index(checkpointloadersimple_4, 0),
                    positive=self.get_value_at_index(cliptextencode_6, 0),
                    negative=self.get_value_at_index(cliptextencode_7, 0),
                    latent_image=self.get_value_at_index(emptylatentimage_5, 0),
                )

                vaedecode_8 = vaedecode.decode(
                    samples=self.get_value_at_index(ksampler_3, 0),
                    vae=self.get_value_at_index(vaeloader_14, 0),
                )

                saveimage_9 = saveimage.save_images(
                    filename_prefix="ComfyUI", images=self.get_value_at_index(vaedecode_8, 0)
                )
                self.image_name = saveimage_9['ui']['images'][0]['filename']

    def display_images(self):
        image_path = os.path.join(self.path, "output", self.image_name)
        image = Image.open(image_path)
        image.show()

    def start(self, batch_size=1):
        image_prompt = self.create_image_prompt()
        self.run(image_prompt, batch_size)
        self.display_images()
        while True:
            prompt = input("What do you want to change?\n")
            self.change_appearance(prompt)
            image_prompt = self.create_image_prompt()
            self.run(image_prompt, batch_size)
            self.display_images()

if __name__ == "__main__":
    pipeline = PersonaImageCreator(persona="jess")
    pipeline.start()
