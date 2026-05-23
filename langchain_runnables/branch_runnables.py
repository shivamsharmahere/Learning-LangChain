from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableSequence, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.5
)

prompt1 = PromptTemplate(
    template="Write a detailed summary about {topic}",
    input_variables=["topic"],
    validate_template=True
)

prompt2 = PromptTemplate(
    template="Write a brief short summary about {text} under 50 words.",
    input_variables=["text"],
    validate_template=True
)

parser = StrOutputParser()

report_gen_chain = RunnableSequence(prompt1, model, parser)
brief_chain = RunnableSequence(prompt2, model, parser)

def inspect_length(text):
    print("report_gen_chain length:", len(text.split()))
    return text
inspect_chain = RunnableLambda(inspect_length)

branch_chain = RunnableBranch(
    (RunnableLambda(lambda x: len(x.split())>50), brief_chain),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain, inspect_chain, branch_chain)
result = final_chain.invoke({"topic": "The impact of climate change on global agriculture."})
print(result)   

final_chain.get_graph().print_ascii()
