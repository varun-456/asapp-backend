from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# FastAPI app setup
app = FastAPI(
    title="Translation API",
    version="1.0",
    description="An API to translate messages using LangChain and Cohere",
)

# 1. Define the model (using Cohere)
model = ChatCohere(api_key="7niJGSzUgsOJFjtvdy6jOhvFK8rp0xnGdR9QVOAI")

# 2. Create the prompt template for translation
system_template = "Translate the following text into {language}:"
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')  # Ensure {text} is expected
])

# 3. Define the message history (sample messages)
messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]

# 4. Define the chain (prompt template, model, and parser)
parser = StrOutputParser()

# Directly connect the prompt, model, and parser without needing 'messages'
chain = prompt_template | model | parser

# Request body model for FastAPI
class TranslationRequest(BaseModel):
    text: str
    language: str

# API Endpoint for translation
@app.post("/translate/")
async def translate(request: TranslationRequest):
    # Prepare the input for the chain
    # Combine recent messages with the new input
    combined_messages = messages + [
        HumanMessage(content=request.text),  # Include the user's text
    ]

    chain_input = {
        "language": request.language,
        "text": request.text  # Pass the correct 'text' variable
    }
    
    # Run the chain and get the response
    response = chain.invoke(chain_input)
    
    return {"translated_text": response}

# Running FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
