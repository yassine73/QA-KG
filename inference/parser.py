from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import TokenTextSplitter
from typing import List
import tempfile

class Parser:
    def __init__(self, file: any):
        self._file_path, self._type = self.get_properties(file)

    def get_properties(self, file):
        if isinstance(file, str):
            return file, file.split('.')[-1].upper()
        elif "streamlit.runtime.uploaded_file_manager.UploadedFile" in str(type(file)):
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.getbuffer())
                return temp_file.name, file.name.split('.')[-1].upper()
        return None

    def load(self) -> List[Document]:
        loader = None
        if self._type == 'PDF':
            loader = PyPDFLoader(file_path=self._file_path)
        elif self._type == 'TXT':
            loader = TextLoader(file_path=self._file_path)
        if loader:
            return loader.load_and_split(TokenTextSplitter(chunk_size=512, chunk_overlap=24))[:1]
        return []