# We will import language specifically tell which type of document we want to load.

from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter, Language
import pymupdf4llm

pdf_path = r"D:\LangChain CampusX\langchain_document_loaders\docs\A Glass Box for the Clinic A Transparent, Dual-LLM Framework for Explainable Medical Report Analysis (2).pdf"

document = pymupdf4llm.to_markdown(pdf_path)

# text_splitter = RecursiveCharacterTextSplitter.from_language(
#     language=Language.MARKDOWN,
#     chunk_size=2000,
#     chunk_overlap=400
# )

#For custom separators
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=400,
    separators=["#", "##", "###", "\n\n", "\n", " ", ""],
)

chunks = text_splitter.split_text(document)
print("Number of chunks created:", len(chunks)) # We will find that the number of chunks created is equal to the number of pages in the PDF divided by the chunk size.
print("First chunk content:", chunks[0]) # The page_content key contains the text of the chunk.