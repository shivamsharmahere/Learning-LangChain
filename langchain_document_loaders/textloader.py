# Text loader is a type of loader used to load text files. It is a simple loader that reads the contents of a text file and returns it as a string.
from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.5
)

prompt = PromptTemplate(
    template="Summarize the following text in 100 words: {text}",
    input_variables=["text"],
    validate_template=True
)

parser = StrOutputParser()
loader = TextLoader("docs/text.txt", encoding="utf-8")

documents = loader.load()
# print(documents)

# print(type(documents)) # We will find that all the docs are loaded in LIST format. Each document is appended to a list.
# print(type(documents[0])) # Each document is in the form of a dictionary. The dictionary has two keys: page_content and metadata.
# print(documents[0].page_content) # The page_content key contains the text of the document.
# print(documents[0].metadata) # The metadata key contains the metadata of the document.

chain = prompt | model | parser
result = chain.invoke({"text": documents[0].page_content})
print("Document Summary:", result)