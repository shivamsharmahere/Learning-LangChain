from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    template="""
You are a {personality} assistant that provides {length} responses in {language}.
When a user asks a question, you should provide a response that is {length} and in {language}.
The response should be in the style of a {personality} assistant.
User Query: {query}
""",
input_variables=["personality", "length", "language", "query"],
validate_template=True
)

template.save("prompt_template.json")