import numpy as np
import streamlit as st
from pinecone import Pinecone
import uuid
import time
from datetime import datetime, timedelta
from transformers import AutoTokenizer, BartForConditionalGeneration
import torch

class HistoryManager():
    def __init__(self, solar_client, pinecone_api_key, dim, device):
        # 임베딩 API client
        self.device = device
        self.client = solar_client
        self.dim = dim
        
        # 대화 내용 저장 인덱스(pinecone)
        index_name = "chatbot"
        self.pc = Pinecone(api_key = pinecone_api_key)

        while not self.pc.describe_index(index_name).status['ready']:
            print("picone index 가져오는 중 ...")
            time.sleep(1)
        
        self.index = self.pc.Index(index_name)

    def get_embedding(self, text):
        embedding = self.client.embeddings.create(
            model = "embedding-query",
            input = text
        ).data[0].embedding

        return embedding

    def find_history_dialogue(self, user_input):
        embedding = self.get_embedding(user_input)

        # 관련 대화 찾기
        related_data = self.index.query(
            vector = embedding,
            top_k = 2,
            include_metadata = True,
            namespace = "summary_data",
        )

        related_dialogue = []
        if related_data:
            related_dialogue = [item.metadata['text'] for item in related_data.matches]

        # 최근 대화 찾기
        stats = self.index.describe_index_stats()
        recent_data_count = stats['namespaces'].get('recent_data',{}).get('vector_count', 0)
        recent_data_count = max(recent_data_count, 1) # 0개일때 오류를 방지하기 위해서
       
        recent_data = self.index.query(
            vector = [0] * self.dim,
            top_k = recent_data_count,
            namespace = "recent_data",
            include_metadata = True
        )

        recent_dialogue = []
        if recent_data:
            recent_dialogue = [item.metadata['text'] for item in recent_data.matches]

        result = {"related_dialogues": related_dialogue, "recent_dialogues": recent_dialogue}
        return result

    def add_history_prompt(self, query, answer):
        dialogue = f"USER : {query} \n\n AIBOT : {answer}"

        embedding = self.get_embedding(dialogue)
        timestamp = datetime.now().isoformat()

        self.index.upsert(vectors=[
            {
                "id": timestamp,
                "values": embedding,
                "metadata": {
                    "text": dialogue,
                    "timestamp": timestamp    
                }
            }
        ], namespace = "recent_data")

        self.manage_data()

    def summarize_text(self, old_data_list):
        tokenizer = AutoTokenizer.from_pretrained("gogamza/kobart-base-v2")
        model = BartForConditionalGeneration.from_pretrained(
            "gogamza/kobart-base-v2",
            device_map="auto",
        )

        input_text = '\n\n'.join(old_data_list)
        print("이전 대화들")
        print(input_text)

        inputs = tokenizer(input_text[:1024], return_tensors="pt").to(self.device)
        print("-" * 100)
        print(inputs)
        print("-" * 100)

        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                max_new_tokens=500,
                num_beams=4,
            )


        output_text = tokenizer.decode(outputs[0], skip_special_tokens = True)

        print("요약 완료")
        print("-" * 1000)
        print(output_text)
        
        return output_text

    def manage_data(self):
        stats = self.index.describe_index_stats()
        recent_data_count = stats['namespaces'].get('recent_data',{}).get('vector_count', 0)
        
        print("누적 최신 메세지 수 : ", recent_data_count)

        if recent_data_count >= 5:
            # 가장 오래된 데이터 5개 뽑아오기
            query_response = self.index.query(
                vector = [0] * self.dim,
                top_k = 5,
                include_metadata = True,
                namespace = "recent_data"
            )
            old_data = [match.metadata['text'] for match in query_response.matches]

            # 오래된 데이터 요약문 만들어서 저장하기 
            summary = self.summarize_text(old_data)
            summary_embedding = self.get_embedding(summary)
            timestamp = datetime.now().isoformat()

            self.index.upsert(
                vectors = [{
                    "id": timestamp,
                    "values": summary_embedding,
                    "metadata": {
                        "text": summary,
                        "timestamp": timestamp
                    }
                }],
                namespace = "summary_data"
            )

            # recent_data 저장소에 오래된 데이터 삭제
            ids_to_delete = [match.id for match in query_response.matches]
            self.index.delete(ids = ids_to_delete, namespace = "recent_data")

    
        
       
    