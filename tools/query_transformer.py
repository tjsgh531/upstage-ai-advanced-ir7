from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

class QueryTransformer():
    def __init__(self, model_name, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            load_in_8bit=True
        )

    def generate_standalone_query(self, chat_prompt):
        inputs = self.tokenizer(chat_prompt, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=50).to(self.device)
        
        search_query = self.tokenizer.decode(output[0], skip_special_tokens=True)
        result = [sent for sent in search_query.split('\n') if sent != '']
        
        return result[-1].strip()

    def create_chat_prompt(self, messages ):
        instruction_message = """
        ## Instructions
        - 당신은 사용자의 여러 대화 메시지를 하나의 검색 쿼리로 변환하는 전문가입니다.
        - 아래 사용자의 대화에서 중요한 정보만 추출하여 간결하고 명확한 검색용 쿼리로 변환하세요.
        - 대화 형식이 아닌, 연구나 조사를 위한 검색어처럼 정확한 질문을 생성하세요.

        ## 대화
        """

        dialogue = [message['content'] for message in messages if message['role'] == 'user']

        prompt = [{"role": "user", "content": instruction_message + '\n'.join(dialogue)}]

        chat_prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

        return chat_prompt
    
    def add_standalone_query(self, queries):
        print("\nstandalone query 생성중 ...")

        results = []
        for query in tqdm(queries):
            messages = query['msg']
            chat_prompt = self.create_chat_prompt(messages)
            standalone_query = self.generate_standalone_query(chat_prompt)
            query["standalone_query"] = standalone_query
            
            results.append(query)

        print("\nstandalone query 생성 완료!")

        return results
