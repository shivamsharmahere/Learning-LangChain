from langchain_groq import ChatGroq
from langchain_core.prompts import load_prompt
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.5
)

st.header("Query Search with Groq and Streamlit")

user_input = st.text_input("Enter your search query:")
length = st.selectbox("Select a length for the response:", ["Concise", "Bulletpoint", "Detailed"])
language = st.selectbox("Select a language for the response:", ["English", "Hindi", "French"])
personality = st.selectbox("Select a personality for the response:", ["Professional", "Casual", "Humorous"])

template = load_prompt("prompt_template.json")

if st.button("Search"):
    if user_input:
        chain = template | model
        response = chain.invoke({
            "personality": personality,
            "length": length,
            "language": language,
            "query": user_input
        })
        st.write(response.content)