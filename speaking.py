from gtts import gTTS
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play

class TextToSpeech:
    def __init__(self, language='pt', tld="com.br", speed=1):
        self.tld = tld
        self.language = language
        self.speed = speed  # Fator de aceleração da voz

    def speak(self, text):
        # Gera o áudio usando gTTS
        tts = gTTS(text=text, lang=self.language, tld=self.tld, slow=False)
        # Salva o áudio em um buffer de bytes em vez de um arquivo
        mp3_fp = BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        # Carrega o áudio do buffer usando pydub
        audio = AudioSegment.from_file(mp3_fp, format="mp3")
        # Ajusta a velocidade de reprodução
        audio_with_changed_speed = audio._spawn(audio.raw_data, overrides={
            "frame_rate": int(audio.frame_rate * self.speed)
        }).set_frame_rate(audio.frame_rate)
        # Reproduz o áudio acelerado
        play(audio_with_changed_speed)

if __name__ == "__main__":
    tts = TextToSpeech(speed=1.2)  # Ajuste o valor de speed conforme desejado
    tts.speak("Olá, meu nome é Hugo e o seu é?")
