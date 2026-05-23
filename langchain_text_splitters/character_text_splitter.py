# CharacterTextSplitter is used to split the text into chunks of a specified size. It is useful when we want to split the text into smaller chunks for processing. For example, if we have a long document and we want to split it into smaller chunks for processing, we can use CharacterTextSplitter.

from langchain_classic.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

pdf_path = "D:\LangChain CampusX\langchain_document_loaders\docs\A Glass Box for the Clinic A Transparent, Dual-LLM Framework for Explainable Medical Report Analysis (2).pdf"

loader = PyPDFLoader(pdf_path)
documents = loader.load()

text_splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    separator=""
)
# Use text_spiltter.split_text to split directly the text of the document.
chunks = text_splitter.split_documents(documents)
print("Total number of documents:", len(documents)) # We will find that the number of documents is equal to the number of pages in the PDF.
print("Number of chunks created:", len(chunks)) # We will find that the number of chunks created is equal to the number of pages in the PDF divided by the chunk size.
print("First chunk content:", chunks[0].page_content) # The page_content key contains the text of the chunk.