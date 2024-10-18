from tools.searchengine import SearchEngine
from tools.query_transformer import QueryTransformer
from tools.generate import Generate
from tools.history_manager import HistoryManager
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
import streamlit as st

import json
import torch

import gc

from dotenv import load_dotenv
import os

class Chatbot():
    def __init__(self):
        self.clean_vram()

        # 환경 변수 설정
        load_dotenv('/upstage-ai-advanced-ir7/.env')
        upstage_api_key = os.getenv('UPSTAGE_API_KEY')
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        openai_api_key = os.getenv('OPENAI_API_KEY')

        # 사용자 설정
        embedding_dim = 4096
        model_name = "rtzr/ko-gemma-2-9b-it"
        self.faiss_path = '/upstage-ai-advanced-ir7/data/faiss_index.faiss'

        # 모델 생성
        # query transfer & generator
        self.is_using_gpt = True
        
        gpt_client = OpenAI(
            api_key = openai_api_key
        )

        if self.is_using_gpt:    
            tokenizer = None
            model = None
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map = "auto",
                load_in_8bit = True
            )

        # solar embedding model
        solar_client = OpenAI(
            api_key = upstage_api_key,
            base_url="https://api.upstage.ai/v1/solar"
        )

        # tool 생성
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.search_engine = SearchEngine(solar_client, embedding_dim, self.device)
        self.query_transformer = QueryTransformer(tokenizer, model, gpt_client, self.device)
        self.history_manager = HistoryManager(solar_client, pinecone_api_key, embedding_dim, self.device)
        self.generator = Generate(tokenizer, model, gpt_client, self.device)

        self.start()

    def start(self):
        with open("/upstage-ai-advanced-ir7/data/documents.jsonl") as f:
            docs = [json.loads(line) for line in f]

        # index 만들기 or 가져오기
        self.search_engine.create_elasticsearch_index(docs)
        self.search_engine.load_faiss_index(self.faiss_path, docs)
        self.clean_vram()

        # chatbot 시작
        # self.chatbot_in_terminal()

        if self.is_using_gpt:
            self.chatbot_in_streamlit_gpt()
        else:
            self.chatbot_in_streamlit()

    def clean_vram(self):
        gc.collect()
        torch.cuda.empty_cache()

    def print_gpu_memory(self):
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                total_memory = torch.cuda.get_device_properties(i).total_memory
                allocated_memory = torch.cuda.memory_allocated(i)
                free_memory = total_memory - allocated_memory
                
                print(f"GPU {i}:")
                print(f"  Total Memory: {total_memory / 1024**3:.2f} GB")
                print(f"  Allocated Memory: {allocated_memory / 1024**3:.2f} GB")
                print(f"  Free Memory: {free_memory / 1024**3:.2f} GB")
                print(f"  Memory Usage: {allocated_memory / total_memory * 100:.2f}%")
        else:
            print("CUDA is not available. GPU memory cannot be accessed.")

    def chatbot_in_terminal(self):
        input_prompt = ""

        while True:
            self.print_gpu_memory()
            input_prompt = input("무엇을 도와드릴까요? : ")
            if input_prompt == 'exit':
                break

            query = [{"role": 'user', "content": input_prompt}]

            chat_prompt = self.query_transformer.create_chat_prompt(query)
            standalone_query = self.query_transformer.generate_standalone_query(chat_prompt)
            print(f"\nstandalone query : {standalone_query}")
            self.clean_vram()

            eval_id = "123"
            query_data = [{"eval_id": eval_id, 'standalone_query': standalone_query}]
            result = self.search_engine.search_queries(query_data)[0]
            self.clean_vram()

            query = result['standalone_query']
            references = result['references']
            response = self.generator.generate_response(query, references)
            self.clean_vram()
            print("\nGemma 답변 : ", response)

    def chatbot_in_streamlit(self):
        st.title("과학 지식 응답 채팅봇")

        # 메세지 기록 초기화
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "과학 지식에 대해 무엇이든 물어보세요!"}]

        # 채팅 기록 표시
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])


        # 사용자 입력
        user_input = st.chat_input("무엇을 도와 드릴까요?")
        
        if user_input:
            # user query 표시 + 이전 대화 표시
            st.session_state.messages.append({"role": "user", "content": user_input})
            with chat_container:
                message = st.session_state.messages[-1]
                print(message)
                with st.chat_message(message["role"]):
                    st.write(message["content"])

            # AI 답변 생성 중 ... 
            with chat_container:
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    message_placeholder.text("AI가 답변을 생성하고 있습니다 ...")

                    # standalone query 생성
                    query = [{"role": "user", "content": user_input}]
                    chat_prompt = self.query_transformer.create_chat_prompt(query)
                    standalone_query = self.query_transformer.generate_standalone_query(chat_prompt)
                    self.clean_vram()

                    # 관련 문서 찾기
                    eval_id = "234"
                    query_data = [{"eval_id": eval_id, "standalone_query": standalone_query}]
                    result = self.search_engine.search_queries(query_data)[0]
                    self.clean_vram()

                    # 과거 대화 찾기
                    history = self.history_manager.find_history_dialogue(user_input)

                    # 답변 생성
                    query = result['standalone_query']
                    references = result['references']
                    response = self.generator.generate_response(query, references, history)
                    self.clean_vram()

                    ai_response_prompt = f"standalone query : \n\n {standalone_query}\n\n 응답 : \n\n {response}"
                    st.session_state.messages.append({"role": "assistant", "content": ai_response_prompt})

                    # 답변 표시
                    message = st.session_state.messages[-1]
                    message_placeholder.empty()
                    st.write(message["content"])
                    print("답변 완료")

                    # 대화 저장
                    self.history_manager.add_history_prompt(user_input, response)
                    
    def chatbot_in_streamlit_gpt(self):
        st.title("과학 지식 응답 채팅봇")

        # 메세지 기록 초기화
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "과학 지식에 대해 무엇이든 물어보세요!"}]

        # 채팅 기록 표시
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

        # 사용자 입력
        user_input = st.chat_input("무엇을 도와 드릴까요?")
        
        if user_input:
            # user query 표시 + 이전 대화 표시
            st.session_state.messages.append({"role": "user", "content": user_input})
            with chat_container:
                message = st.session_state.messages[-1]
                print(message)
                with st.chat_message(message["role"]):
                    st.write(message["content"])

            # AI 답변 생성 중 ... 
            with chat_container:
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    message_placeholder.text("AI가 답변을 생성하고 있습니다 ...")

                    # 과거 대화 찾기
                    history = self.history_manager.find_history_dialogue(user_input)

                    # standalone query 생성
                    standalone_query = self.query_transformer.create_standalone_query_gpt(user_input)
                    if standalone_query: # 과학 관련 질문이면
                        # 관련 문서 찾기
                        eval_id = "234"
                        query_data = [{"eval_id": eval_id, "standalone_query": standalone_query}]
                        result = self.search_engine.search_queries(query_data)[0]
                        query = result['standalone_query']
                        self.clean_vram()

                        # 관련문서 참고하여 답변 생성
                        references = result['references']
                        response = self.generator.generate_response_gpt(query, references, history)
                    else:
                        print("과학 관련 질문이 아닙니다")
                        response = self.generator.generate_non_science_gpt(user_input, history)
                 
                    ai_response_prompt = f"standalone query : \n\n {standalone_query}\n\n 응답 : \n\n {response}"
                    st.session_state.messages.append({"role": "assistant", "content": ai_response_prompt})

                    # 답변 표시
                    message = st.session_state.messages[-1]
                    message_placeholder.empty()
                    st.write(message["content"])
                    print("답변 완료")

                    # 대화 저장
                    self.history_manager.add_history_prompt(user_input, response)

if __name__ == '__main__':
    gc.collect()
    torch.cuda.empty_cache()
    Chatbot()
