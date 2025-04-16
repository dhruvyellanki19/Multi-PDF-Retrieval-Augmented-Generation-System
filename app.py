import os
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# --- Load API Key from .env ---
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


# --- Page Config and Background ---
st.set_page_config(page_title="DocuMind AI", page_icon="🧠", layout="wide")

def add_bg_from_url():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://images.unsplash.com/photo-1525186402429-1eef5b15f9ec?auto=format&fit=crop&w=1400&q=80");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_url()

# --- Styled Title ---
st.markdown("""
    <div style="padding: 2rem; background-color: rgba(255,255,255,0.85); border-radius: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
        <h1 style="text-align: center; font-size: 3rem; color: #1c1c1c;">🧠 DocuMind AI</h1>
        <h3 style="text-align: center; color: #333;">Multi-PDF AI ChatBot Agent</h3>
        <p style="text-align: center; font-size: 1.2rem; color: #444;">
            Interact with multiple documents using the power of AI. Upload. Ask. Learn.
        </p>
    </div>
""", unsafe_allow_html=True)


# --- Function to process uploaded PDFs ---
def loadPDF(files):
    try:
        st.write("📄 Processing uploaded PDFs...")
        if not os.path.exists("pdf"):
            os.mkdir("pdf")
        for i, file in enumerate(files):
            with open(f"pdf/temp{i+1}.pdf", "wb") as f:
                f.write(file.getvalue())

        loader = PyPDFDirectoryLoader("pdf")
        documents = loader.load()
        st.write(f"✅ Loaded {len(documents)} documents from uploaded PDFs.")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        split_docs = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings()
        vectorstore = FAISS.from_documents(split_docs, embedding=embeddings)
        vectorstore.save_local("vectordb")
        st.success("✅ Vector store created and saved.")
        return True

    except Exception as e:
        st.error(f"❌ Error while processing PDFs: {e}")
        return False

    finally:
        for f in os.listdir("pdf"):
            try:
                os.remove(os.path.join("pdf", f))
            except:
                pass

# --- Function to create the QA chain ---
def get_chain():
    retriever = FAISS.load_local("vectordb", HuggingFaceEmbeddings(), allow_dangerous_deserialization=True).as_retriever()

    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful assistant knowledgeable about the uploaded PDF documents.

        Use ONLY the provided context to answer.
        If you don’t know the answer, say: "I'm sorry, I couldn't find the answer in the uploaded documents."

        <context>
        {context}
        </context>

        Question: {input}

        Answer:
        """
    )

    # ✅ Using supported Groq model
    llm = ChatGroq(model="llama3-8b-8192")
    doc_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, doc_chain)


# --- Sidebar File Upload ---
st.sidebar.image("image.png")
st.sidebar.title("📁 Upload Files Here")
uploaded_files = st.sidebar.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if st.sidebar.button("Upload & Process"):
    if uploaded_files:
        st.info("📦 Starting PDF processing...")
        if loadPDF(uploaded_files):
            st.session_state["chain"] = get_chain()
            st.success("🚀 AI ChatBot is ready! Ask your questions below.")
        else:
            st.error("PDF processing failed.")
    else:
        st.warning("⚠️ Please upload at least one PDF file.")


# --- Chat Section ---
if "chain" in st.session_state:
    question = st.text_input("Ask your question:")
    if question:
        with st.spinner("💡 Thinking..."):
            response = st.session_state["chain"].invoke({"input": question})
            st.write(response["answer"])

st.sidebar.markdown("---")
st.sidebar.write("Project by [Dhruv Yellanki](https://github.com/dhruvyellanki19)")
