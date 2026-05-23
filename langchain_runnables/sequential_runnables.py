from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.5
)

prompt1 = PromptTemplate(
    template="Write a short brief about the following topic: {topic}",
    input_variables=["topic"],
    validate_template=True
)

prompt2 = PromptTemplate(
    template="Give me 2 MCQ questions about the following brief: {brief}",
    input_variables=["brief"],
    validate_template=True
)

parser = StrOutputParser()

chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

result = chain.invoke({"topic": "The impact of climate change on polar bears"})
print("MCQ Questions:\n", result)
