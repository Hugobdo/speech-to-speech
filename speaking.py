from gtts import gTTS
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play
import os
import torch
import se_extractor
from api import BaseSpeakerTTS, ToneColorConverter

class TextToSpeech:
    def __init__(self, language='pt', tld="com.br", speed=1, custom_base=True, voice='example_reference.mp3', ckpt_path='checkpoints/converter', ckpt_base='checkpoints/base_speakers/EN'):
        self.tld = tld
        self.language = language
        self.speed = speed
        self.custom_base = custom_base
        self.output_dir = "outputs"
        self.base_file = f"{self.output_dir}/base.mp3"
        self.reference_file = f"resources/{voice}"
        self.output_path = fr'{self.output_dir}/output_crosslingual.wav'
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if not custom_base:
            self.base_speaker_tts = BaseSpeakerTTS(f'{ckpt_base}/config.json', device=self.device)
            self.base_speaker_tts.load_ckpt(f'{ckpt_base}/checkpoint.pth')
            self.source_se = torch.load(f'{ckpt_base}/en_default_se.pth').to(self.device)

        self.tone_color_converter = ToneColorConverter(f'{ckpt_path}/config.json', device=self.device)
        self.tone_color_converter.load_ckpt(f'{ckpt_path}/checkpoint.pth')

    def gtts_speaker(self, text, play_audio=True, save=False, filename="output.mp3", lang=None, tld=None, speed=None):
        if lang is None:
            lang = self.language
        if tld is None:
            tld = self.tld
        if speed is None:
            speed = self.speed

        tts = gTTS(text=text, lang=lang, tld=tld, slow=False)
        mp3_fp = BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        audio = AudioSegment.from_file(mp3_fp, format="mp3")
        audio_with_changed_speed = audio._spawn(audio.raw_data, overrides={
            "frame_rate": int(audio.frame_rate * speed)
        }).set_frame_rate(audio.frame_rate)

        if save:
            audio_with_changed_speed.export(filename, format="mp3")
        
        if play_audio:
            play(audio_with_changed_speed)
    
    def create_base_file(self):
        input_text = "Este áudio será utilizado para extrair a incorporação da tonalidade base da voz do locutor. " + \
        "Normalmente, um áudio muito curto deve ser suficiente, mas aumentar o comprimento do áudio " + \
        "também melhorará a qualidade do áudio de saída."
        self.gtts_speaker(input_text, play_audio=False, save=True, filename=self.base_file)
            
    def convert_tone_color(self, text, encode_message="@MyShell"):
        src_path = f'{self.output_dir}/tmp.wav'

        if self.custom_base:
            if not os.path.exists(self.base_file):
                print("Base file not found. Creating a new one.")
                self.create_base_file()
            self.gtts_speaker(text, play_audio=False, save=True, filename=src_path)
            self.source_se, _ = se_extractor.get_se(self.base_file, self.tone_color_converter, vad=True)
        
        else:
            self.base_speaker_tts.tts(text, src_path, speaker='default', language='English', speed=self.speed)
        
        self.target_se, _ = se_extractor.get_se(self.reference_file, self.tone_color_converter, vad=True)

        self.tone_color_converter.convert(
            audio_src_path=src_path, 
            src_se=self.source_se, 
            tgt_se=self.target_se, 
            output_path=self.output_path,
            message=encode_message)
        
    def play_audio(self):
        """
        Reproduz o áudio a partir de um arquivo especificado.

        Args:
        file_path (str): Caminho para o arquivo de áudio a ser reproduzido.
        """
        audio = AudioSegment.from_file(self.output_path)
        play(audio)
    
    def speak(self, text):
        self.convert_tone_color(text)
        self.play_audio()

if __name__ == "__main__":
    tts = TextToSpeech(custom_base=False, speed=1)

    text = "Hey honey. I'm going to the store. Do you need anything?"
    tts.speak(text)
