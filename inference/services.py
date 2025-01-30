from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain.base_language import BaseLanguageModel
from typing import List

class HybridService:
    def __init__(self, llm: BaseLanguageModel):
        self.graph_client = Neo4jGraph(refresh_schema=False)
        self.vector_client = None
        self.chain = GraphCypherQAChain.from_llm(
            graph=self.graph_client, llm=llm, verbose=True, allow_dangerous_requests=True
        )
    
    def add_graph(self, documents: List[GraphDocument]):
        self.graph_client.add_graph_documents(documents, include_source=True, baseEntityLabel=True)
        return 200, "success"
    
    def add_vector(self, documents: List[Document]):
        self.vector_client.add_documents(documents)
        return 200, "success"
    
    def query_graph(self, query: str):
        return self.chain.invoke(query)