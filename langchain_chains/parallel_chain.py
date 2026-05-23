from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

model1 = ChatGroq(
    model="openai/gpt-oss-20b", #this model support structured output
    temperature=0.5
)

parser = StrOutputParser()

template1 = PromptTemplate(
    template="Give me a brief summary and notes on {text}.",
    input_variables=["text"],
    validate_template=True
)

template2 = PromptTemplate(
    template="Create a quiz with 2 question and 4 options on this text {text}.",
    input_variables=["text"],
    validate_template=True
)

template3= PromptTemplate(
    template="Create a single documemt with summary, notes and quiz on {notes}, {quiz}",
    input_variables=["notes", "quiz"],
    validate_template=True
)

parallel_chain = RunnableParallel(
    {"notes": template1 | model1 | parser,
     "quiz": template2 | model1 | parser}
)

merge_chain = template3 | model1 | parser

chain = parallel_chain | merge_chain

text = """Logistic Regression is a supervised machine learning algorithm used for classification problems. Unlike linear regression, which predicts continuous values it predicts the probability that an input belongs to a specific class.

It is used for binary classification where the output can be one of two possible categories such as Yes/No, True/False or 0/1.
It uses sigmoid function to convert inputs into a probability value between 0 and 1.
_what_is_logistic_regression.webp_what_is_logistic_regression.webp
Types of Logistic Regression
Logistic regression can be classified into three main types based on the nature of the dependent variable:

Binomial Logistic Regression: This type is used when the dependent variable has only two possible categories. Examples include Yes/No, Pass/Fail or 0/1. It is the most common form of logistic regression and is used for binary classification problems.
Multinomial Logistic Regression: This is used when the dependent variable has three or more possible categories that are not ordered. For example, classifying animals into categories like "cat," "dog" or "sheep." It extends the binary logistic regression to handle multiple classes.
Ordinal Logistic Regression: This type applies when the dependent variable has three or more categories with a natural order or ranking. Examples include ratings like "low," "medium" and "high." It takes the order of the categories into account when modeling."""
result = chain.invoke({"text": text})

print("Summary:", result)
chain.get_graph().print_ascii()

