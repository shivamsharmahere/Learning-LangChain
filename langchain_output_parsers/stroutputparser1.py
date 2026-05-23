from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-20b", #this model support structured output
    temperature=0.5
)

template1 = PromptTemplate(
    template="Give me a brief summary on this {topic}.",
    input_variables=["topic"],
    validate_template=True
)

template2 = PromptTemplate(
    template="Give me a 5 lines brief summary on this on this text {text}.",
    input_variables=["text"],
    validate_template=True
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic": "Python programming"})
print("Summary:", result)