# WebBaseloader is a loader for the WebBase dataset, which is a large collection of web pages. The loader reads the dataset from a specified directory and yields documents in a format suitable for processing by language models.
# It is mostly used for static web pages having more HTML and less text. It can be used to extract text from web pages and create Document objects that can be used in LangChain. Each web page is treated as a separate document, and the metadata for each document includes the URL of the web page.

from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.5
)

parser = StrOutputParser()

prompt = PromptTemplate(
    template = "What is the main topic of the following web page? {text}",
    input_variables=["text"],
    validate_template=True
)

url = "https://rasa.com/docs/pro/tutorial/"
loader = WebBaseLoader(url)

documents = list(loader.lazy_load())

# print("Number of documents loaded:", len(documents)) # We will find that the number of documents loaded is equal to the number of web pages in the dataset.
# print("First document content:", documents[0].page_content) # The page_content key contains the text of the document.
# print("First document metadata:", documents[0].metadata) # The metadata key contains the metadata

chain = prompt | model | parser
result = chain.invoke({"text": documents[0].page_content})
print("Main topic of the web page:", result)