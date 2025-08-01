from attr import validate
from openai import OpenAI

# similar to response.py, this file handles a final validation over the response
# currently limited to follow the structure of the response

class ValidationAgent():
    """
    Validates structured final answer from retrieved information.
    """

    def __init__(self, openai_api_key):
        self.client = OpenAI(api_key=openai_api_key)
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

        Please provide your validated answer."""

        # validated response
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


