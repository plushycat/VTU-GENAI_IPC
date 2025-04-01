import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Initialize Groq LLM
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
llm = ChatGroq(model_name="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)

# Load Hugging Face embedding model
hf_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

# Function to create FAISS index
def create_faiss_index(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_text(text)
    embeddings = hf_model.encode(texts)
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(np.array(embeddings))
    return index, texts

# Function to retrieve relevant IPC section
def retrieve_ipc_section(query, index, chunks):
    query_embedding = hf_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k=1)
    return chunks[indices[0][0]] if indices[0][0] < len(chunks) else "No relevant section found."

# Define the prompt template
prompt = PromptTemplate(
    input_variables=["ipc_section", "query"],
    template="""
    You are an expert in Indian law. A user asked: "{query}"
    Based on the Indian Penal Code (IPC), the relevant section is:
    {ipc_section}
    
    Please provide:
    - A simple explanation
    - The key legal points
    - Possible punishments
    - A real-world example
    """
)

# Function to interact with the chatbot
def ipc_chatbot(query, index, chunks):
    related_section = retrieve_ipc_section(query, index, chunks)
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(ipc_section=related_section, query=query)
    return response

# Streamlit UI
st.image("PragyanAI_Transperent.png")
st.title("IPC Legal Chatbot")

uploaded_file = st.file_uploader("Upload IPC PDF", type=["pdf"])
if uploaded_file:
    ipc_text = extract_text_from_pdf(uploaded_file)
    ipc_faiss_index, ipc_chunks = create_faiss_index(ipc_text)
    st.success("PDF processed successfully!")

query = st.text_area("Enter your legal question:")
if st.button("Get Answer"):
    if uploaded_file and query:
        response = ipc_chatbot(query, ipc_faiss_index, ipc_chunks)
        st.write("### Response:")
        st.write(response)
    else:
        st.error("Please upload an IPC PDF and enter a query.")
