# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import nest_asyncio
import re
from pyngrok import ngrok
from config import FEW_SHOT_PROMPT, NGROK_AUTH_TOKEN
from langchain.prompts import PromptTemplate
from embeddings import setup_qa_chain

app = FastAPI()
nest_asyncio.apply()

# Initialize components
llm = setup_qa_chain().llm
retriever = setup_qa_chain().retriever

# Models
class DoctorQuestion(BaseModel):
    message: str
    translated_conversation: str

# Prompt template
template = PromptTemplate(
    template=FEW_SHOT_PROMPT,
    input_variables=["translated_conversation", "raag_reference", "question"]
)

def clean_text(text):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    text = ansi_escape.sub('', text).strip()
    text = re.sub(r'Prompt after formatting:.*?\n', '', text, flags=re.DOTALL)
    match = re.search(r'Answer:\s*(.*)', text, re.DOTALL)
    answer = match.group(1).strip() if match else text
    return "\n".join(dict.fromkeys(answer.split("\n")))

@app.get("/")
def root():
    return {"message": "Dermatology Assistant API with RAAG is running."}

@app.post("/ask")
def ask(msg: DoctorQuestion):
    user_input = msg.message.strip()
    translated_conversation = msg.translated_conversation.strip()

    try:
        # Retrieve medical references
        retrieved_docs = retriever.get_relevant_documents(user_input)
        raag_context = "\n".join([doc.page_content for doc in retrieved_docs])

        # Prepare prompt
        prompt = template.format(
            translated_conversation=translated_conversation,
            raag_reference=raag_context,
            question=user_input
        )

        # Send to LLM
        result = llm(prompt)
        cleaned = clean_text(result)
        return {"response": cleaned}
    except Exception as e:
        return {"error": str(e)}

def run_server():
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    public_url = ngrok.connect(8000)
    print("Your API is available at:", public_url)
    
    import uvicorn
    uvicorn.run(app, port=8000)

if __name__ == "__main__":
    run_server()
