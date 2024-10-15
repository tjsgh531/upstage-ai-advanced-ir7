import faiss
import numpy as np
import streamlit as st

class HistoryManager():
    def __init__(self, solar_client, dim):
        # 대화내용 저장
        self.client = solar_client
        st.session_state.dialogue_history = []
        self.index = faiss.IndexFlatL2(dim)

    def find_history_dialogue(self, user_input):
        dialogue_history = st.session_state.dialogue_history

        # 유사한 대화
        query_embedding = self.client.embeddings.create(
            model = "embedding-query",
            input = user_input
        ).data[0].embedding

        query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
        distances, indices = self.index.search(query_embedding, 2)

        similar_dialogue = []
        for idx in indices[0]:
            # 비어있거나 유효한 idx가 없을 때 faiss는 -1을 반환한데요
            if idx == -1:
                continue
            
            sim_dialogue = dialogue_history[idx]
            similar_dialogue.append(sim_dialogue)

        # 최근 대화
        current_dialogue = []
        for temp_dialogue in dialogue_history[-2:]:
            if temp_dialogue not in similar_dialogue:
                current_dialogue.append(temp_dialogue)

        return {"current_dialogue" : current_dialogue, "sim_dialogue" : similar_dialogue}
    
    def add_history_prompt(self, query, answer):
        dialogue_history = st.session_state.dialogue_history
        dialogue = f"user : {query} \n\n model : {answer}"
        
        # dialogue list에 추가
        dialogue_history.append(dialogue)
        
        # fiass 에 추가
        dialogue_embedding = self.client.embeddings.create(
            model = "embedding-query",
            input = dialogue
        ).data[0].embedding

        dialogue_embedding = np.array([dialogue_embedding]).astype('float32')
        self.index.add(dialogue_embedding)
    