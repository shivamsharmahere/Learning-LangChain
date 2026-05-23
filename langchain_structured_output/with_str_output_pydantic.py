from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import List, Optional, TypedDict, Annotated, Literal
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-20b", #this model support structured output
    temperature=0.5
)

class Review(BaseModel):
    summary: str = Field(..., description="A brief summary of the review")
    rating: float = Field(...,gt=0.0, lt=10.0, description="A float rating between 1.0 and 10.0")
    pros: List[str] = Field(..., description="A list of pros")
    cons: List[str] = Field(..., description="A list of cons")
    author: Optional[str] = Field(None, description="The name of the review author")
    sentiment: Literal["positive", "negative", "neutral"] = Field(..., description="The overall sentiment of the review")

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""The hardware is great, but the software feels bloated. There are
too many pre-installed apps that I can't remove. Also, the UI looks outdated compared to
other brands. Hoping for a software update to fix this.""")

print(result)
result_dict = result.model_dump_json()
print(result_dict)