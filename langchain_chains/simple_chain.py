from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-20b", #this model support structured output
    temperature=0.5
)

template = PromptTemplate(
    template="Give me a brief summary on this in 100-120 words:  {topic}.",
    input_variables=["topic"],
    validate_template=True
)

parser = StrOutputParser()

chain = template | model | parser

result = chain.invoke({"topic": "Python programming"})
print("Summary:", result)
chain.get_graph().print_ascii()