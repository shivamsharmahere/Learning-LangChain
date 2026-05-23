from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

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

prompt1 = template1.invoke({"topic": "Python programming"})
result1 = model.invoke(prompt1)

prompt2 = template2.invoke({"text": result1})
result2 = model.invoke(prompt2)

print("Summary:", result1.content)
print("*" * 5 + " 5 Lines Summary " + "*" * 5)
print("5 lines summary:", result2.content)