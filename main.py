from tools.searchengine import SearchEngine
from tools.query_transformer import QueryTransformer
from tools.generate import Generate

import json
import torch

import gc

from dotenv import load_dotenv
import os

# ElasticSearch 실행
# sudo -u daemon -s bash -c "bash /upstage-ai-advanced-ir7/elasticsearch-8.8.0/bin/elasticsearch"

class Main:
    def __init__(self):
        # VRAM 초기화
        self.clean_vram()

        # 환경 변수 설정
        load_dotenv('/upstage-ai-advanced-ir7/.env')
        upstage_api_key = os.getenv('UPSTAGE_API_KEY')
        os.environ["OPENAI_API_KEY"] = upstage_api_key
        

        # 사용자 설정
        embedding_dim = 4096
        query_transformer_model_name = "rtzr/ko-gemma-2-9b-it"
        generate_model_name = "rtzr/ko-gemma-2-9b-it"

        self.output_filename = "./outputs/sample_submission.csv"
        self.faiss_path = '/upstage-ai-advanced-ir7/data/faiss_index.faiss'

        # tool 생성
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.search_engine = SearchEngine(embedding_dim, upstage_api_key)
        self.query_transformer = QueryTransformer(query_transformer_model_name, device)
        self.generator = Generate(generate_model_name, device)

        # 실행
        self.start()


    def start(self):
        # docs 데이터 가져오기
        with open("./data/documents.jsonl") as f:
            docs = [json.loads(line) for line in f]

        # eval 데이터 가져오기
        with open("./data/eval.jsonl") as f:
            eval_data = [json.loads(line) for line in f]

        # 색인 만들기 & docs 추가
        self.search_engine.create_elasticsearch_index(docs)
        # self.search_engine.create_faiss_index(self.faiss_path, docs) # 처음이면 create_faiss
        self.search_engine.load_faiss_index(self.faiss_path, docs) # create_faiss_index로 만든 적있으면 load
        self.clean_vram()

        # standalone query 추가
        standalone_query_eval_data = self.query_transformer.add_standalone_query(eval_data[:5])
        self.clean_vram()

        # 검색
        ir_results = self.search_engine.search_quries(standalone_query_eval_data)
        self.clean_vram()

        # query + 검색 => 답변 생성
        self.generator.generate_bulk_response(ir_results)
        self.clean_vram()
        
        # 검색 결과 파일에 저장
        self.write_results(ir_results, self.output_filename)

    def write_results(self, results, output_filename):
        with open(output_filename, 'w') as of:
            for result in results:
                of.write(f"{json.dumps(result, ensure_ascii=False)}\n")

    def clean_vram(self):
        gc.collect()
        torch.cuda.empty_cache()

if __name__ =="__main__":
    Main()