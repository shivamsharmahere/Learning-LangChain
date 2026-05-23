from langchain_groq import ChatGroq
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-20b", #this model support structured output
    temperature=0.5
)

class PersonInfo(BaseModel):
    name : str = Field(..., description="The name of the person")
    age: int = Field(..., gt=18, description="The age of the person, must be greater than 18")
    city: str = Field(..., description="The city where the person lives")

parser = PydanticOutputParser(pydantic_object=PersonInfo)

template = PromptTemplate(
    template= "Give me the name , age , city of a fictional {place} person \n {format_instructions}",
    input_variables=["place"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
    validate_template=True
)

chain = template | model | parser
result = chain.invoke({"place": "New York"})
print("Parsed Result:", result)