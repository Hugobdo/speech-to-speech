import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
from PIL import Image
import json
import threading
import queue

class PersonaImageCreator:
    def __init__(self, persona):
        self.path = r"C:\Users\T-GAMER\Desktop\Projetos\ComfyUI_windows_portable\ComfyUI"
        self.persona = persona
        with open(f"personas/{self.persona}.json", "r") as f:
            data = json.load(f)
            self.original_appearance = data['appearance']
            self.model = data.get('model', 'hassakuHentaiModel_v13.safetensors')
        self.image_queue = queue.Queue()
        self.generation_done = False  # Indica se a geração de imagens foi concluída

    def create_prompt(self):
        with open(f"personas/{self.persona}.json", "r") as f:
            appearance = json.load(f)['appearance']
            
        prompt = ""
        for key, value in appearance.items():
            prompt += f"{value}, "
        return prompt

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
                self.image_queue.put(saveimage_9['ui']['images'][0]['filename'])
            self.generation_done = True

    def display_images(self):
        while not (self.generation_done and self.image_queue.empty()):
            try:
                image_name = self.image_queue.get(timeout=10)  # Espera por uma imagem ou timeout
                image_path = os.path.join(self.path, "output", image_name)
                image = Image.open(image_path)
                image.show()
                self.image_queue.task_done()
            except queue.Empty:
                # Se a fila estiver vazia e a geração concluída, sai do loop
                break

    def start(self, prompt, batch_size=1):
        # Método para iniciar as threads de geração e exibição de imagens
        generation_thread = threading.Thread(target=self.run, args=(prompt, batch_size))
        display_thread = threading.Thread(target=self.display_images)
        
        generation_thread.start()
        display_thread.start()

        generation_thread.join()  # Aguarda a conclusão da thread de geração

if __name__ == "__main__":
    pipeline = PersonaImageCreator(persona="jess")
    
    prompt = pipeline.create_prompt()
    pipeline.start(prompt, batch_size=2)
