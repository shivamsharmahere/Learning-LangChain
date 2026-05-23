# Recursive Character Text Splitter is a text splitter that splits text into chunks based on a specified separator. It recursively splits the text

from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser

pdf_path = "D:\LangChain CampusX\langchain_document_loaders\docs\A Glass Box for the Clinic A Transparent, Dual-LLM Framework for Explainable Medical Report Analysis (2).pdf"

parser = StrOutputParser()

loader = PyPDFLoader(pdf_path)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=400,
    separators=["\n\n", "\n", " ", ""]
)

chunks = text_splitter.split_documents(documents)
print("Total number of documents:", len(documents)) # We will find that the number of documents is equal to the number of pages in the PDF.
print("Number of chunks created:", len(chunks)) # We will find that the number of chunks created is equal to the number of pages in the PDF divided by the chunk size.
print("First chunk content:", chunks[0].page_content) # The page_content key contains the text of the chunk.