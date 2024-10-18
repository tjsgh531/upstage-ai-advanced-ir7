import torch
from tqdm import tqdm
from openai import OpenAI
import numpy as np
import faiss

class QueryTransformer():
    def __init__(self, tokenizer, model, gpt_client, device):
        self.tokenizer = tokenizer
        self.model = model
        self.client = gpt_client
        self.device = device
        

    def generate_standalone_query(self, chat_prompt):
        chat_template_prompt = self.tokenizer.apply_chat_template(
            chat_prompt,
            tokenize = False,
        )

        inputs = self.tokenizer(
            chat_template_prompt,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=300)

        standalone_query = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]: ],
            skip_special_tokens = True
        )

        return standalone_query

    def create_chat_prompt(self, messages):
        instruction_message = """
        ## Instructions
        - 당신은 사용자의 여러 대화 메시지를 하나의 검색 쿼리로 변환하는 전문가입니다.
        - 아래 사용자의 대화에서 중요한 정보만 추출하여 간결하고 명확한 검색용 쿼리로 변환하세요.
        - 대화 형식이 아닌, 연구나 조사를 위한 검색어처럼 정확한 질문을 생성하세요.

        ## 대화
        """

        dialogue = [{"role": message["role"], "content": message["content"]} for message in messages]
        chat_prompt = [{"role": "user", "content": instruction_message}]
        chat_prompt.extend(dialogue)

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

    def create_standalone_query_gpt(self, user_input):
        system_message = {
            "role": "system",
            "content": """
                당신은 과학적 또는 학문적 논의와 관련된 질문에 답변하는 전문가입니다.
                질문이 전혀 과학적이지 않은 경우에만 '과학 관련 질문이 아닙니다.'라고 답변하세요.
                질문이 과학과 조금이라도 아주 조금이라도 관련이 있다면, 예를 들어 음식, 여가 등 매우 기초적인 것이라도 그 질문을 구체적이고 명확한 검색 쿼리로 변환하세요.
                과학적 논의는 자연과학, 사회과학, 기술, 심리학, 역사적 연구 등 다양한 학문 분야를 포함할 수 있습니다.
                단, 변환된 쿼리를 명확히 출력하되, 변환 과정을 설명하거나 예시로 출력하지 말고, **변환된 쿼리만 출력하세요.**
                
                예를 들어:
                - '복잡한 데이터 구조 설계 방법을 알려줘.' -> '복잡한 데이터 구조 설계 방법'
                - '인간의 감정이 사회적 관계에 미치는 영향' -> '인간의 감정이 사회적 관계에 미치는 영향'
                변환된 쿼리만 출력하세요.
            """
        }

        user_message = {
            "role" : 'user',
            "content" : user_input
        }

        # 사용자 질문 준비
        prompt = [system_message, user_message]

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=prompt,
            temperature=0
        )

        standalone_query = response.choices[0].message.content 

        if "과학 관련 질문이 아닙니다" in standalone_query:
            return False
        else:
            return standalone_query