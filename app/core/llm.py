import os
from dotenv import load_dotenv
from openai import OpenAI

class SiliconFlowLLM():
    def __init__(self):
        load_dotenv()
        self.SILICON_FLOW_API_KEY = os.getenv("SILICON_FLOW_API_KEY")
        self.SILICON_FLOW_BASE_URL= os.getenv("SILICON_FLOW_BASE_URL")
        
        # models
        self.SILICON_FLOW_REASONING_MODEL = os.getenv("SILICON_FLOW_REASONING_MODEL")
        self.SILICON_FLOW_NL2SQL_MODEL = os.getenv("SILICON_FLOW_NL2SQL_MODEL")
        self.SILICON_FLOW_HELPER_MODEL = os.getenv("SILICON_FLOW_HELPER_MODEL")
        print("-" * 50)
        print(f"""\ncoder llm: {self.SILICON_FLOW_NL2SQL_MODEL}\nreasoning llm: {self.SILICON_FLOW_REASONING_MODEL}\nhelper llm: {self.SILICON_FLOW_HELPER_MODEL}\n""")
        print("-" * 50)

        # client
        self.client = OpenAI(api_key=self.SILICON_FLOW_API_KEY,
                             base_url=self.SILICON_FLOW_BASE_URL)
        
    def call_coder(self, query, prompt):
        response = self.client.chat.completions.create(
            model=self.SILICON_FLOW_NL2SQL_MODEL,  
            messages=[    
                {"role": "system", "content": prompt},  
                {"role": "user", "content": query}  
            ],  
            temperature=0.7,  
            max_tokens=4096  
        )  
        content = response.choices[0].message.content
        return content
    
    def call_llm(self, query, prompt):
        response = self.client.chat.completions.create(
            model=self.SILICON_FLOW_REASONING_MODEL,  
            messages=[    
                {"role": "system", "content": prompt},  
                {"role": "user", "content": query}  
            ],  
            temperature=0.7,  
            max_tokens=4096  
        )  
        content = response.choices[0].message.content
        return content

    def call_helper(self, query, prompt):
        response = self.client.chat.completions.create(
            model=self.SILICON_FLOW_HELPER_MODEL,  
            messages=[    
                {"role": "system", "content": prompt},  
                {"role": "user", "content": query}  
            ],  
            temperature=0.7,  
            max_tokens=4096  
        )  
        content = response.choices[0].message.content
        return content
    