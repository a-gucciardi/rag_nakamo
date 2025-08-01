from openai import OpenAI

# similar to response.py, this file handles a final validation over the response
# currently limited to follow the structure of the response

class ValidationAgent():
    """
    Validates structured final answer from retrieved information.
    """
    def __init__(self, client="OpenAI", openai_api_key=None, hf_model="google/gemma-3-4b-it", ollama_model="llama3.1:8b"):
        self.client_type = client
        if client == "OpenAI":
            self.client = OpenAI(api_key=openai_api_key)

        elif client == "HF":
            from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
            import torch, os
            os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                hf_model, 
                device_map="auto", 
                torch_dtype=torch.bfloat16,
                attn_implementation="eager"
            ).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(hf_model)

        elif client == "Ollama":
            import ollama
            self.ollama_model = ollama_model
            self.ollama = ollama
        self.message_history = []

    def process_message(self, message):
        #  RESPONSE STRUCTURE:
        # - Executive Summary (brief overview)
        # - Detailed Analysis (main content with citations)
        # - Key Requirements/Standards (if applicable)
        # - Sources (list all referenced documents)

        # -> parse the search results AFTER final response agent
        self.message_history.append(message)

        # message_dict = self.as_dict(message)
        # print(message_dict)
        system_prompt = """
        You are a regulatory expert assistant. Charged with the validation of the previous expert assistant reponse. 
        Your task is to promake sure the answers are in the scope of regulatory questions about medical devices.
        Then simply copy the comprehensive, accurate previous answers to regulatory questions about medical devices based on the provided regulatory documents.
        IMPORTANT GUIDELINES:
        1. Preserve the structure of the response and its subcontent and titles
        2. Keep the sources and citations, verify relevance
        3. Verify individually the sections using professional, technical language appropriate for regulatory context
        4. Remove excessive confidence or unnecessary information
        5. Copy the whole thing with your corrections if needed

        RESPONSE STRUCTURE (as input):
        - ## Executive Summary (brief overview)
        - ## Detailed Analysis (main content with citations)
        - ## Key Requirements/Standards (if applicable)
        - ## Sources (list all referenced documents)

        Keep citation format: [Source Name, Page X] after each major point."""

        user_prompt = f"""
        Previous response: {message}

        Please correct the answer if needed."""

        # validated response
        if self.client_type == "OpenAI":
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            validated_message = response.choices[0].message.content

        elif self.client_type == "HF":
            import torch
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
                ]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            with torch.inference_mode():
                outputs = self.model.generate(**inputs, max_new_tokens=1500, do_sample=False)
            validated_message = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

        elif self.client_type == "Ollama":
            response = self.ollama.chat(model=self.ollama_model, messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            validated_message = response['message']['content']

        # print(validated_message)
        return validated_message


    def as_dict(self, message):
        # message -> dict 
        titles = ["Executive Summary", "Detailed Analysis", "Key Requirements/Standards", "Sources"]
        result = {}
        current_title = None
        buffer = []

        for line in message.splitlines():
            line = line.strip()
            if line.startswith("##"):
                heading = line[2:].strip()
                if heading in titles:
                    if current_title:
                        result[current_title] = "\n".join(buffer).strip()
                    current_title = heading
                    buffer = []
            else:
                if current_title:
                    buffer.append(line)

        if current_title:
            result[current_title] = "\n".join(buffer).strip()

        return result


