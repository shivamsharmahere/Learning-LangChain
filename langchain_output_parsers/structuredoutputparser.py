#Structured Output Parser Deprecated

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-20b", #this model support structured output
    temperature=0.5
)

parser = StructuredOutputParser()

schema = [
    ResponseSchema(name="Fact1: ", description="A fact about the topic"),
    ResponseSchema(name="Fact2: ", description="Another fact about the topic"),
    ResponseSchema(name="Fact3: ", description="Yet another fact about the topic")
]

template = PromptTemplate(
    template="Give me 3 facts about {topic} \n {format_instructions}",
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions(schema)},
    validate_template=True
)

chain = template | model | parser
result = chain.invoke({"topic": "Python programming"})
print("Structured Output:", result)
