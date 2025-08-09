from openai import OpenAI

class ResponseAgent():
    """
    Creates structured final answers from retrieved information.
    Uses LLM to synthesize information and format responses with proper citations.
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
        """
        Process retrieved information and generate a structured response
        """
        self.message_history.append(message)

        # -> parse the search results AFTER RAG agent
        query = message.get("query", "")
        results = message.get("results", [])
        original_question = message.get("original_question", query)

        # Generate structured response using LLM
        # structured_response = self._generate_structured_response(original_question, results)
        context_sections = []
        sources = []
        for result in results:
            context_sections.append(f"""
            Source: {result['source']} (Page: {result['page']})
            Content: {result['content']}
            Document Type: {result['document_type']}
            """)
            source_ref = f"{result['source']} (Page: {result['page']})"
            if source_ref not in sources:
                sources.append(source_ref)
        source_list = "\n\nSources Referenced:\n" + "\n".join([f"- {source}" for source in sources])

        context = "\n---\n".join(context_sections)
        system_prompt = """
        You are a regulatory expert assistant. 
        Your task is to provide comprehensive, accurate answers to regulatory questions about medical devices based on the provided regulatory documents.
        IMPORTANT GUIDELINES:
        1. Base your answer ONLY on the provided regulatory documents
        2. Provide a structured response with clear sections
        3. Include specific citations for each major point
        4. If the documents don't contain enough information, clearly state this
        5. Use professional, technical language appropriate for regulatory context
        6. Highlight key requirements, processes, or standards mentioned
        7. Compare FDA vs WHO approaches when relevant

        RESPONSE STRUCTURE:
        - ## Executive Summary (brief overview)
        - ## Detailed Analysis (main content with citations)
        - ## Key Requirements/Standards (if applicable)
        - ## Sources (list all referenced documents)

        Use citation format: [Source Name, Page X] after each major point."""

        user_prompt = f"""
        Regulatory Question: {original_question}

        Available Regulatory Information:
        {context}

        Please provide a comprehensive, structured answer based on the regulatory documents provided above."""

        # final response
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
            structured_response = response.choices[0].message.content

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
            structured_response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

        elif self.client_type == "Ollama":
            response = self.ollama.chat(model=self.ollama_model, messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            structured_response = response['message']['content']

        # we add source list at the end
        return structured_response + source_list
