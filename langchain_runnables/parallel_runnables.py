from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableParallel, RunnableSequence
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.5
)
prompt1 = PromptTemplate(
    template="Generate a tweet about {topic}",
    input_variables=["topic"]
)   

prompt2 = PromptTemplate(
    template="Generate a LinkedIn post about {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "tweet": prompt1 | model | parser,
    "linkedin_post": prompt2 | model | parser
})

result = parallel_chain.invoke({"topic": "the benefits of AI in healthcare"})
print("Generated Tweet:\n", result["tweet"])
print("\nGenerated LinkedIn Post:\n", result["linkedin_post"])