from typing import List
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_groq import ChatGroq
from langchain_community.llms.replicate import Replicate

from .parser import Parser
from .services import HybridService

llm = Replicate(
    # model="meta/meta-llama-3-70b-instruct",
    model="meta/meta-llama-3.1-405b-instruct",
    model_kwargs={"temperature": 0.1, "top_p": 0.1, "max_tokens": 1024},
)
service = HybridService(llm=llm)

class HybridAgent:
    def __init__(self, **kwargs):
        self.file = kwargs.get('file', None)
        
        if not self.file:
            raise ValueError('File is required')
        if not llm:
            raise ValueError('LLM model is required')
        if not service:
            raise ValueError('Service is required')
        
        self.document = self.load_document()
        self.llm_transformer = LLMGraphTransformer(llm)
        self.process_file()

        
    def load_document(self):
        parser = Parser(file=self.file)
        document = parser.load()
        return document
    
    def process_file(self):
        graph_documents = self.llm_transformer.convert_to_graph_documents(documents=self.document)
        code, message = service.add_graph(graph_documents)
        
        if code != 200:
            raise ValueError(f'Error processing file: {message}')
        
        
    
    
    def ask(self, messages: List[any]):
        return service.query_graph(messages[-1].content)
        