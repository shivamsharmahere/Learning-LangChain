from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda

load_dotenv()   

model = ChatGroq(
    model="openai/gpt-oss-20b", #this model support structured output
    temperature=0.5
)

parser1 = StrOutputParser()

class SentimentAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = Field(..., description="The sentiment of the review")

parser2 = PydanticOutputParser(pydantic_object=SentimentAnalysis)

prompt = PromptTemplate(
    template="Classify the sentiment of this review as positive, negative or neutral: {feedback} \n {format_instructions}",
    input_variables=["feedback"],
    partial_variables={"format_instructions": parser2.get_format_instructions()},
    validate_template=True
)

classifier_chain = prompt | model | parser2

prompt1 = PromptTemplate(
    template= "Write a positive response to this feedback: {feedback}",
    input_variables=["feedback"],
    validate_template=True
)


branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt1 | model | parser1),
    (lambda x: x.sentiment == 'negative', prompt1 | model | parser1),
    (lambda x: x.sentiment == 'neutral', prompt1 | model | parser1),
    RunnableLambda(lambda x: "Invalid sentiment")
)

conditional_chain = classifier_chain | branch_chain

result = conditional_chain.invoke({"feedback": "The product is terrible!"})
print("Response:", result)

conditional_chain.get_graph().print_ascii()

