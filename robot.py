from hearing import AudioHandler
from reasoning import MainChatModel
from speaking import TextToSpeech

class Robot:
    def __init__(self, model_name="base", samplerate=16000, verbose=True):
        self.hearing = AudioHandler(model_name=model_name, samplerate=samplerate)
        self.reasoning = MainChatModel()
        self.speaking = TextToSpeech(custom_base=False, speed=1.1)
        self.verbose = verbose
        self.speaking.speak("Hello, I am your personal assistant. How can I help you?")

    def listen_and_respond(self):
        if self.verbose:
            print("Ouvindo...")
        question_audio = self.hearing.record_audio()
        if self.verbose:
            print("Transcrevendo áudio...")
        question_text = self.hearing.transcribe_audio(question_audio)
        if question_text.lower() == " you":
            return

        if question_text.lower() == "stop":
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
