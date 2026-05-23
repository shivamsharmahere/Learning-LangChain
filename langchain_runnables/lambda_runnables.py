from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

def count_words(text: str) -> int:
    return len(text.split())

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.5
)

prompt1 = PromptTemplate(
    template = "Write a joke about {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

joke_generation_chain = prompt1 | model | parser

parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "word_count": RunnableLambda(count_words)
})

final_chain = RunnableSequence(joke_generation_chain, parallel_chain)
result = final_chain.invoke({"topic": "programming"})
print("Generated Joke:\n", result["joke"])
print("\nWord Count of the Joke:\n", result["word_count"])