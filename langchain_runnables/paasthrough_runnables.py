#Paasthrough Runnables allow you to pass the output of one chain to another chain without modification. This is useful when you want to use the same output in multiple places or when you want to perform different operations on the same output.
# Eg = parser.RunnablePaasthrough(5) will output 5
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.5
)

prompt1 = PromptTemplate(
    template = "Write a joke about {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template = "Write the explanation of the joke: {joke}",
    input_variables=["joke"]
)

parser = StrOutputParser()

joke_generation_chain = prompt1 | model | parser

parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(), # This will take the output of joke_generation_chain and pass it to the next chain
    "explanation": RunnableSequence(prompt2, model, parser)
})

final_chain = RunnableSequence(joke_generation_chain, parallel_chain)

result = final_chain.invoke({"topic": "programming"})
print("Generated Joke:\n", result["joke"])
print("\nExplanation of the Joke:\n", result["explanation"])