# CSV loader is used to load CSV files. It reads the contents of a CSV file and returns it as a list of dictionaries, where each dictionary represents a row in the CSV file with the column names as keys.
# It gives us document for each row in the CSV file.

from langchain_community.document_loaders import CSVLoader
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
    template = "What is the name of the employee in the following row? {text}",
    input_variables=["text"],
    validate_template=True
)

csv_path = "docs/employees.csv"
loader = CSVLoader(csv_path)
documents = loader.load()
# print(len(documents)) # We will find that the number of documents is equal to the

chain = prompt | model | parser
result = chain.invoke({"text": documents[0].page_content})
print("Name of the employee in the first row:", result)
