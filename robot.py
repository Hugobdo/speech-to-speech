from hearing import AudioHandler
from reasoning import LocalChatModel
from speaking import TextToSpeech  # ou TextToSpeechPyttsx3, se preferir

class Robot:
    def __init__(self, model_name="base", samplerate=16000, language='pt', tld="com.br", verbose=True):
        self.hearing = AudioHandler(model_name=model_name, samplerate=samplerate)
        self.reasoning = LocalChatModel()
        self.speaking = TextToSpeech(language=language, tld=tld)  # Inicializa o TTS
        self.verbose = verbose

    def listen_and_respond(self):
        if self.verbose:
            print("Ouvindo...")
        question_audio = self.hearing.record_audio()
        if self.verbose:
            print("Transcrevendo áudio...")
        question_text = self.hearing.transcribe_audio(question_audio)
        if question_text.lower() == " you":
            return
        
        if self.verbose:
            print(f"Pergunta: {question_text}")
            print("Pensando na resposta...")
        answer = self.reasoning.ask(question_text)
        if self.verbose:
            print(f"Resposta: {answer}")
            print("Respondendo...")
        self.speaking.speak(answer)  # Usa TTS para vocalizar a resposta

if __name__ == "__main__":
    robot = Robot(verbose=True)
    while True:
        robot.listen_and_respond()
