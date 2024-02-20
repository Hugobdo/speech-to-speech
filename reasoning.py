from openai import OpenAI

class MainChatModel:
    def __init__(self, base_url="http://localhost:1234/v1", api_key="not-needed", model="local-model"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.system_prompt = {
            "role": "system",
            "content": '''
                Your name is HX3.
                You are friendly and always respond casually to the best of your ability.
                You are brief and direct, but always polite.
                Always respond in a way that sounds as human as possible.
                Keep all the sentences very short
            '''
        }
        self.memory = []  # Inicializa a memória para armazenar interações

    def ask(self, user_message, temperature=0.7):
       
        messages = [self.system_prompt] + self.memory + [{"role": "user", "content": user_message}]
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=100
            )
            
            self.memory.append({"role": "user", "content": user_message})
            self.memory.append({"role": "assistant", "content": completion.choices[0].message.content})
            
            return completion.choices[0].message.content

        except Exception as e:
            print(f"Erro ao solicitar resposta do modelo: {e}")
            return None

    def clear_memory(self):
        self.memory.clear()

if __name__ == "__main__":
    model = MainChatModel()
    response = model.ask("Quem é você?")
    print(response)

    response = model.ask("Me conte mais sobre você.")
    print(response)
