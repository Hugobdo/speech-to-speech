from hearing import AudioHandler
from reasoning import MainChatModel
from speaking import TextToSpeech

class Robot:
    def __init__(self, model_name="base", samplerate=16000, verbose=True):
        self.hearing = AudioHandler(model_name=model_name, samplerate=samplerate)
        self.reasoning = MainChatModel()
        self.speaking = TextToSpeech(custom_base=False, speed=1, voice='example_reference.mp3')
        self.verbose = verbose
        self.speaking.speak("Hello, I am your personal assistant. How can I help you?")

    def listen_and_respond(self):
        while True:  # Continue tentando até que o áudio com som seja gravado
            if self.verbose:
                print("Ouvindo...")
            question_audio = self.hearing.record_audio()
            # Verifica se o áudio contém som antes de prosseguir

            if not self.hearing.check_audio_silence(question_audio):
                break
            elif self.verbose:
                print("Nenhum som detectado. Pode falar...")

        if self.verbose:
            print("Transcrevendo áudio...")
        question_text = self.hearing.transcribe_audio(question_audio)

        # stop if question contains stop
        if question_text.lower().split(" ")[-1].split('.')[0] == "stop":
            self.speaking.speak("See you later!")
            exit()
        
        if self.verbose:
            print(f"Pergunta: {question_text}")
            print("Pensando na resposta...")
        answer = self.reasoning.ask(question_text)
        if self.verbose:
            print(f"Resposta: {answer}")
            print("Respondendo...")
        self.speaking.speak(answer)

if __name__ == "__main__":
    robot = Robot(verbose=True)
    while True:
        robot.listen_and_respond()
