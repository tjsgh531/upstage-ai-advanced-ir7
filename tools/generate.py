import torch
from tqdm import tqdm

class Generate():
    def __init__(self, tokenizer, model, gpt_client, device):
        self.device = device
        self.tokenizer = tokenizer
        self.model = model
        self.client = gpt_client

    def create_text(self, query, references, history):
        # 지시 사항
        instruction_message = f"""
        ## Role
        - 과학 상식 전문가
        
        ## Instructions
        - 주어진 참조 문헌 정보를 활용하여 질문에 대해 간결하게 답변을 생성한다.
        - 주어진 참조 문헌 정보로 대답할 수 없는 경우는 정보가 부족해서 답을 할 수 없다고 대답한다.
        - 최근 대화를 참조하여 답변을 명확하게 생성할 수 있으면 최근 대화를 참조하여 대답한다.
        - 관련 대화를 참조하여 답변을 명확하게 생성할 수 있으면 관련 대화를 참조하여 대답한다.
        - 여러가지를 복합적으로 참조하여도 된다.
        - 무엇을 참조하였는지 밝히면서 대답한다.

        ## 질문
        {query}

        """

        # 참조 문헌
        reference_messages = "\n"
        for idx, ref in enumerate(references):
            reference_messages += f"""
            ## 참조 문헌 {idx}
            {ref["content"]}

            """

        #과거 대화
        recent_dialogue = '\n\n'.join(history['recent_dialogues'])
        related_dialogue = '\n\n'.join(history['related_dialogues'])
        history_dialogue = f"""

        ## 관련 대화
        {related_dialogue}

        ## 최근 대화
        {recent_dialogue}

        """

        input_text = instruction_message + reference_messages + history_dialogue + "\n ## 답변 \n" 
        
        print(input_text)
        return input_text
    
    def create_text_gpt(self, query, references, history):
        # 지시 사항
        instruction_message = f"""
        ## Role
        - 당신은 과학 상식 전문가 입니다.
        
        ## Instructions
        - 주어진 참조 문헌 정보를 활용하여 질문에 대해 간결하게 답변을 생성한다.
        - 주어진 참조 문헌 정보로 대답할 수 없는 경우는 정보가 부족해서 답을 할 수 없다고 대답한다.
        - 최근 대화를 참조하여 답변을 명확하게 생성할 수 있으면 최근 대화를 참조하여 대답한다.
        - 관련 대화를 참조하여 답변을 명확하게 생성할 수 있으면 관련 대화를 참조하여 대답한다.
        - 여러가지를 복합적으로 참조하여도 된다.
        - 무엇을 참조하였는지 밝히면서 대답한다.
        """

        # 참조 문헌
        reference_messages = ""
        for idx, ref in enumerate(references):
            reference_messages += f"""
            ## 참조 문헌 {idx}
            {ref["content"]}

            """

        #과거 대화
        recent_dialogue = '\n\n'.join(history['recent_dialogues'])
        related_dialogue = '\n\n'.join(history['related_dialogues'])
        history_dialogue = f"""

        ## 관련 대화
        {related_dialogue}

        ## 최근 대화
        {recent_dialogue}

        """

        system_message = {"role": "system", "content": instruction_message + reference_messages + history_dialogue}
        user_message = {"role":"user", "content": query}

        input_text = [system_message,  user_message]
        
        print(input_text)
        return input_text

    def create_text_non_science_gpt(self, query, history):
        # 지시 사항
        instruction_message = f"""
        ## Instructions
        - 최근 대화를 참조하여 답변을 명확하게 생성할 수 있으면 최근 대화를 참조하여 대답한다.
        - 관련 대화를 참조하여 답변을 명확하게 생성할 수 있으면 관련 대화를 참조하여 대답한다.
        - 무엇을 참조하였는지 밝히면서 대답한다.
        """

        #과거 대화
        recent_dialogue = '\n\n'.join(history['recent_dialogues'])
        related_dialogue = '\n\n'.join(history['related_dialogues'])
        history_dialogue = f"""

        ## 관련 대화
        {related_dialogue}

        ## 최근 대화
        {recent_dialogue}

        """

        system_message = {"role": "system", "content": instruction_message + history_dialogue}
        user_message = {"role":"user", "content": query}

        input_text = [system_message,  user_message]
        
        print(input_text)
        return input_text

    def generate_response(self, query, references, history):
        input_text = self.create_text(query, references, history)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=300)

        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]: ], skip_special_tokens = True)
        return response

    def generate_response_gpt(self, query, references, history):
        input_text = self.create_text_gpt(query, references, history)

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages = input_text
        )

        output = response.choices[0].message.content
        return output

    def generate_non_science_gpt(self, query, history):
        input_text = self.create_text_non_science_gpt(query, history)

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages = input_text
        )

        output = response.choices[0].message.content
        return output
        


    def generate_bulk_response(self, ir_results):
        results = []
        for result in tqdm(ir_results):
            query = result["standalone_query"]
            references = result["references"]

            response = self.generate_response(query, references)
            result['answer'] = response
            results.append(result)

        return results

