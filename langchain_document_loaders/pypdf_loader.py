# PyPDFloader is a document loader that uses the PyPDF library to load PDF documents. 
# It can be used to extract text from PDF files and create Document objects that can be used in LangChain of each page of the PDF. 
# Each page is treated as a separate document, and the metadata for each document includes the page number and the source file name.

from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.5
)

parser = StrOutputParser()

prompt = PromptTemplate(
    template = "Who are the authors of the following document? {text}",
    input_variables=["text"],
    validate_template=True
)

pdf_path = r"docs/A Glass Box for the Clinic A Transparent, Dual-LLM Framework for Explainable Medical Report Analysis (2).pdf"
loader = PyPDFLoader(pdf_path)

documents = loader.load()
# print(len(documents)) # We will find that the number of documents is equal to the number of pages in the PDF.
# print(type(documents)) # We will find that all the docs are loaded in LIST format. Each document is appended to a list.
# print(type(documents[0])) # Each document is in the form of a dictionary. The dictionary has two keys: page_content and metadata.
# print(documents[0].page_content) # The page_content key contains the text of the document.
# print(documents[0].metadata) # The metadata key contains the metadata of the document.

chain = prompt | model | parser
result = chain.invoke({"text": documents[0].page_content})
print("Authors of the document:", result)
