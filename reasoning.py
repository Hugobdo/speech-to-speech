from openai import OpenAI
import json

class MainChatModel:
    def __init__(self, base_url="http://localhost:1234/v1", api_key="not-needed", model="local-model", system_prompt=None, temperature=0.7, max_tokens=200):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = model
        if system_prompt is not None:
            self.content = system_prompt
        else:
            self.content = "You are friendly and always respond casually to the best of your ability. You are brief and direct, but always polite. Always respond in a way that sounds as human as possible. Keep all the sentences very short."
        
        self.system_prompt = {
            "role": "system",
            "content": f"{self.content}"
        }
        self.memory = []

    def ask(self, user_message):
        messages = [self.system_prompt] + self.memory + [{"role": "user", "content": user_message}]
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            self.memory.append({"role": "user", "content": user_message})
            self.memory.append({"role": "assistant", "content": completion.choices[0].message.content})
            
            return completion.choices[0].message.content

        except Exception as e:
            print(f"Erro ao solicitar resposta do modelo: {e}")
            return None

    def change_sd_prompt(self, appearance, prompt):
        self.system_prompt = {
            "role": "system",
            "content": "You are an expert on image generation with AI. You will receive json inputs and a message, then will outpud the new json, changing whatever is needed to create what the user expects to see. You don't talk, you don't explain, you only outputs json"
        }
        messages = [self.system_prompt] + [{"role": "user", "content": f"{appearance}"}] + [{"role": "user", "content": prompt}]
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        result = completion.choices[0].message.content
        json_data = self.parse_json(result)
        print(f"JSON data: {json_data}")
        return json_data
        
    def parse_json(self, input_string):
        # Encontrando o índice inicial do JSON
        input_string = input_string.replace("'", '"')
        start_index = input_string.find('{')
        # Encontrando o índice final do JSON
        end_index = input_string.rfind('}') + 1
        
        if start_index != -1 and end_index != -1:
            # Extraindo a substring que é o JSON
            json_str = input_string[start_index:end_index]
            try:
                # Convertendo a string para um dicionário
                json_dict = json.loads(json_str)
                return json_dict
            except json.JSONDecodeError as e:
                print(f"Erro ao decodificar o JSON: {e}")
                return None
        else:
            print("Não foi possível encontrar um JSON na string fornecida.")
            return None

    def clear_memory(self):
        self.memory.clear()

if __name__ == "__main__":
    # model = MainChatModel()
    # response = model.ask("Quem é você?")
    # print(response)

    sd_chat = MainChatModel(temperature=0, max_tokens=-1)
    with open("personas/jess.json", "r") as f:
        data = json.load(f)
        appearance = data['appearance']

    prompt = sd_chat.change_sd_prompt(appearance, "become a redhead")
