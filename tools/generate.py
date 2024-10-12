from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoConfig
import torch
from tqdm import tqdm

class Generate():
    def __init__(self, model_name, device):
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        quantization_config = QuantoConfig(weights="int8")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map = "auto",
            load_in_8bit = True
            )

    def create_text(self, query, references):
        instruction_message = f"""
        ## Role
        - 과학 상식 전문가
        
        ## Instructions
        - 주어진 참조 문헌 정보를 활용하여 질문에 대해 간결하게 답변을 생성한다.
        - 주어진 참조 문헌 정보로 대답할 수 없는 경우는 정보가 부족해서 답을 할 수 없다고 대답한다.
        - 주어진 참조 문헌 정보를 복합적으로 사용하여 답변을 생성 할 수 있다.
        - 어떤 참조 문헌을 참조했는지 명확히 밝히면서 답변을 생성한다.
        - 한국어로 답변을 생성한다.

        ## 질문
        {query}

        """

        reference_messages = "\n"
        for idx, ref in enumerate(references):
            reference_messages += f"""
            ## 참조 문헌 {idx}
            {ref["content"]}

            """

        return instruction_message + reference_messages + "\n ## 답변 \n"

    def generate_response(self, query, references):
        input_text = self.create_text(query, references)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=300)

        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]: ], skip_special_tokens = True)
        return response

    def generate_bulk_response(self, ir_results):
        results = []
        for result in tqdm(ir_results):
            query = result["standalone_query"]
            references = result["references"]

            response = self.generate_response(query, references)
            result['answer'] = response
            results.append(result)

        return results

