from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import List, Optional, TypedDict, Annotated, Literal

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-20b", #this model support structured output
    temperature=0.5
)

class Review(TypedDict):
    summary: Annotated[str, "A brief summary of the review"]
    rating: Annotated[float, "A float rating between 1.0 and 10.0"]
    pros: Annotated[List[str], "A list of pros"]
    cons: Annotated[List[str], "A list of cons"]
    author: Annotated[Optional[str], "The name of the review author"]
    sentiment: Annotated[Literal["positive", "negative", "neutral"], "The overall sentiment of the review"]

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""The hardware is great, but the software feels bloated. There are
too many pre-installed apps that I can't remove. Also, the UI looks outdated compared to
other brands. Hoping for a software update to fix this.""")

print(result)