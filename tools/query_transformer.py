import torch
from tqdm import tqdm
from openai import OpenAI
import numpy as np
import faiss

class QueryTransformer():
    def __init__(self, tokenizer, model, device):
        self.tokenizer = tokenizer
        self.model = model
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
