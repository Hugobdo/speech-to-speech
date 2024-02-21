import numpy as np
import sounddevice as sd
import whisper
import threading
import queue

class AudioHandler:
    def __init__(self, model_name="base", samplerate=16000, verbose=False):
        self.model = whisper.load_model(model_name)
        self.verbose = verbose
        self.samplerate = samplerate
        self.audio_queue = queue.Queue()  # Fila para armazenar blocos de áudio para transcrição

    def play_audio(self, audio):
        if self.verbose:
            print("Reproduzindo áudio...")
        sd.play(audio, self.samplerate)
        sd.wait()

    def transcribe_audio(self, audio):
        result = self.model.transcribe(audio)
        if self.verbose:
            print(result["text"])
        return result["text"]

    def record_audio(self):
        if self.verbose:
            print("Gravando áudio, fale agora...")
        block_duration = 0.1  # Duração de cada bloco de áudio em segundos
        silence_threshold = 0.5  # Threshold de volume para considerar silêncio
        silence_duration = 0  # Duração acumulada de silêncio
        recording_blocks = []  # Armazena blocos de gravação
        recording_active = True  # Controla se a gravação está ativa

        def callback(indata, frames, time, status):
            nonlocal silence_duration, recording_active
            if status:
                print(status)
            volume_norm = np.linalg.norm(indata) * 10
            if volume_norm < silence_threshold:
                silence_duration += block_duration
                if silence_duration >= 2:  # 2 segundos de silêncio
                    recording_active = False  # Sinaliza para parar a gravação
            else:
                silence_duration = 0  # Reset silêncio
            recording_blocks.append(indata.copy())

        with sd.InputStream(callback=callback, blocksize=int(self.samplerate * block_duration),
                            channels=1, samplerate=self.samplerate):
            while recording_active:  # Continua gravando enquanto ativo
                sd.sleep(int(block_duration * 1000))  # Aguarda um bloco de duração para verificar novamente

        recording = np.concatenate(recording_blocks, axis=0)
        if self.verbose:
            print("Gravação concluída.")
        return recording.flatten()

    def check_audio_silence(self, audio, silence_threshold=15):
        volume_norm = np.linalg.norm(audio)*10
        if self.verbose:
            print(f"Volume Normalizado: {volume_norm:.4f}, Limiar de Silêncio: {silence_threshold}")
        return volume_norm < silence_threshold

    def record_continuous_audio(self, block_duration=0.5, silence_threshold=0.5):
        def callback(indata, frames, time, status):
            if status:
                print(status)
            volume_norm = np.linalg.norm(indata) * 10
            if volume_norm > silence_threshold:
                self.audio_queue.put(indata.copy())

        with sd.InputStream(callback=callback, blocksize=int(self.samplerate * block_duration),
                            channels=1, samplerate=self.samplerate):
            if self.verbose:
                print("Gravando... Pressione Ctrl+C para parar.")
            try:
                while True:
                    sd.sleep(100)
            except KeyboardInterrupt:
                if self.verbose:
                    print("Gravação finalizada.")

    def transcribe_from_queue(self):
        while True:
            if not self.audio_queue.empty():
                audio_block = self.audio_queue.get()
                if audio_block is not None:
                    self.transcribe_audio(audio_block.flatten())
                self.audio_queue.task_done()

    def start_recording_and_transcription(self):
        threading.Thread(target=self.record_continuous_audio, daemon=True).start()
        threading.Thread(target=self.transcribe_from_queue, daemon=True).start()

        # Mantém o programa principal rodando, caso contrário os threads daemon terminarão imediatamente
        try:
            while True:
                sd.sleep(1000)
        except KeyboardInterrupt:
            if self.verbose:
                print("Encerrando gravação e transcrição.")

if __name__ == "__main__":
    audio_handler = AudioHandler(model_name='medium', verbose=True)
    audio = audio_handler.record_audio()
    audio_handler.transcribe_audio(audio)
