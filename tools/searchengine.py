import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import heapq
import torch
from elasticsearch import Elasticsearch, helpers
from openai import OpenAI
from tqdm import tqdm

class SearchEngine:
    def __init__(self, solar_client, dim, device):
        self.device = device
        self.index = faiss.IndexFlatL2(dim)
        self.client = solar_client
        self.documents = []

        self.es_index_name = "es_index"

        # reranking 토크나이저 & 모델
        self.tokenizer = AutoTokenizer.from_pretrained("Dongjin-kr/ko-reranker")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "Dongjin-kr/ko-reranker",
            device_map="auto",
            load_in_4bit = True
        ) 
    
    # elastic
    def create_elasticsearch_index(self, docs):
        print("\nelastic search client 생성중 ...")

        es_username = 'elastic' 
        es_password = 'Yu10AEV5Kj7vhGBRxGeJ'
        self.es = Elasticsearch(
            ['https://localhost:9200'],
            basic_auth=(es_username, es_password),
            ca_certs="/upstage-ai-advanced-ir7/elasticsearch-8.8.0/config/certs/http_ca.crt"
        )

        settings = {
            "analysis": {
                "analyzer" : {
                    "nori": {
                        "type" : "custom",
                        "tokenizer" : "nori_tokenizer",
                        "decompound_mode" : "mixed",
                        "filter" : ["nori_posfilter"]
                    }
                },
                "filter": {
                    "nori_posfilter": {
                        "type": "nori_part_of_speech",
                        "stoptags" : ["E", "J", "SC", "SE", "SF", "VCN", "VCP", "VX"]
                    }
                }
            }
        }

        mappings = {
            "properties" : {
                "content": {"type": "text", "analyzer": "nori"},
                "embeddings" : {
                    "type": "dense_vector",
                    "dims": 768,
                    "index": True,
                    "similarity": "l2_norm"
                }
            }
        }

        # index가 없을때 만 생성하자
        if not self.es.indices.exists(index = self.es_index_name):
            self.es.indices.create(index = self.es_index_name, settings = settings, mappings = mappings)

            actions = [
                {
                    '_index' : self.es_index_name,
                    '_source' : doc
                }
                for doc in docs
            ]

            helpers.bulk(self.es, actions)

        print("\nelastic search client 생성 완료 ... ")

    # elastic
    def elastic_search(self, query_str, k=50):
        query = {
            "match" : {
                "content" : {
                    "query" : query_str
                }
            }
        }

        results = self.es.search(
            index = self.es_index_name,
            query = query,
            size = k,
            sort = "_score"
        )

        references = []
        for rst in results['hits']['hits']:
            references.append({
                "docid": rst['_source']['docid'], 
                "score": float(rst['_score']),
                "content": rst['_source']['content']})
        
        return references

    # faiss
    def create_faiss_index(self, faiss_path, docs):
        print("\nfaiss index 생성중 ...")

        embeddings = []
        for doc in tqdm(docs):
            content = doc["content"]
            embedding = self.client.embeddings.create(
                model = "embedding-passage",
                input = content
            ) .data[0].embedding

            embeddings.append(embedding)

        embeddings = np.array(embeddings).astype('float32')

        self.index.add(embeddings)
        self.documents.extend(docs)

        faiss.write_index(self.index, faiss_path)

        print("\nfaiss index 생성 완료!")

    def load_faiss_index(self, faiss_path, docs):
        print("\nfaiss index load 중 ...")
        self.index = faiss.read_index(faiss_path)
        self.documents.extend(docs)
        print("\nfaiss index load 완료!")

    # faiss
    def faiss_search(self, query:str, k=50):
        query_embedding = self.client.embeddings.create(
            model = "embedding-query",
            input = query
        ).data[0].embedding

        query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
        distances, indices = self.index.search(query_embedding, k)

        references = [{
            "docid": self.documents[i]["docid"],
            "score": float(distances[0][idx]),
            "content": self.documents[i]["content"]
            } for idx, i in enumerate(indices[0]) if idx != -1]

        return references

    # reranking
    def exp_normalize(self, x):
        b = x.max()
        y = np.exp(x - b)
        return y / y.sum()

    # reranking
    def reranker(self, query, docs):
        pairs = [[query, doc['content']] for doc in docs]
        self.model.eval()

        with torch.no_grad():
            inputs = self.tokenizer(pairs, padding = True, truncation = True, return_tensors = 'pt', max_length=512).to(self.device)
            scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
            scores = self.exp_normalize(scores.cpu().numpy())
            indicies = [(-score, idx) for idx, score in enumerate(scores)]
            heapq.heapify(indicies)

        # top3 뽑기
        topk = []
        references = []
        for _ in range(3):
            score, idx = heapq.heappop(indicies)

            docid = docs[idx]["docid"]
            reference = {"score": float(-score), "content": docs[idx]['content']}

            topk.append(docid)
            references.append(reference)

        return topk, references

    # main
    def search_queries(self, queries):
        results = []

        print("검생 중 ...")
        for query in tqdm(queries):
            eval_id = query["eval_id"]
            standalone_query = query['standalone_query']
            
            faiss_references = self.faiss_search(standalone_query)
            es_references = self.elastic_search(standalone_query)

            refs_ids = {}
            refs = []
            for faiss_ref, es_ref in zip(faiss_references, es_references):
                faiss_id = faiss_ref["docid"]
                es_id = es_ref["docid"]

                if faiss_id not in refs_ids:
                    refs_ids[faiss_id] = 1
                    refs.append(faiss_ref)
                
                if es_id not in refs_ids:
                    refs_ids[es_id] = 1
                    refs.append(es_ref)

            print(f"query와 연관된 문서 개수{len(refs)}")
            topk = []
            references = []
            if len(refs) > 0:
                topk, references = self.reranker(standalone_query, refs)

            results.append({
                "eval_id" : eval_id,
                "standalone_query": standalone_query,
                "topk": topk,
                "references": references
            })
        
        return results
    
    