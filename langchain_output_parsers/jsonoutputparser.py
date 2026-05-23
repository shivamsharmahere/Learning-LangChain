from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-20b", #this model support structured output
    temperature=0.5
)

parser = JsonOutputParser()

template = PromptTemplate(
    template= "Give me the name , age , city of a fictional person \n {format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()},
    validate_template=True
)

# prompt = template.format()
# result = model.invoke(prompt)
# parsed_result = parser.parse(result.content)    
# print("Parsed Result:", parsed_result)

#Now using chains

chain = template | model | parser
result = chain.invoke({})
print("Parsed Result:", result)