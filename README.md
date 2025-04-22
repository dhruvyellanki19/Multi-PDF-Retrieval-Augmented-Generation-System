# DocuMind AI: Multi-PDF ChatBot Agent 🧠


**DocuMind AI** is an AI-powered application that allows users to upload multiple PDF documents, process them, and interact with the content through an intelligent chatbot. Powered by **Langchain**, **Groq**, and **FAISS**, this app enables users to query their PDFs and receive contextually relevant responses in real time.

## Features

- **Multi-PDF Upload**: Upload up to 10 PDF files at once.
- **Instant AI-Powered Responses**: Ask questions based on the uploaded documents and get context-based answers.
- **FAISS Vector Search**: Efficiently retrieves document sections using embeddings for quick and accurate results.
- **Generative AI Model**: Uses **Groq** and **Llama3** models for real-time AI responses.
- **Interactive Chat Interface**: A user-friendly interface built with **Streamlit**.
- **Upload History**: Track your previous uploads and clear history as needed.
- **Environment Variables Support**: Use `.env` files for secure storage of API keys.

## Description

DocuMind AI offers a simple and effective way to interact with multiple PDF documents. Users can upload one or more PDF files, and the app will process the documents, extracting text and creating embeddings for fast retrieval. Once the PDFs are processed, the AI chatbot can respond to questions based on the uploaded documents.

## How It Works

1. **PDF Upload**: Users upload one or more PDFs through the sidebar.
2. **Text Extraction**: The app extracts text from the PDFs using **PyPDF2** and splits it into smaller chunks for efficient querying.
3. **Embeddings**: The text is converted into embeddings using **HuggingFaceEmbeddings**, allowing for faster search and retrieval.
4. **FAISS Retrieval**: The embeddings are stored in a **FAISS** vector store, enabling quick similarity searches.
5. **Query Processing**: Users can ask questions, and the AI retrieves the most relevant document sections to generate an answer.
6. **AI Response Generation**: Using the **Groq** model and **Llama3**, the AI generates responses based on the retrieved document content.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/dhruvyellanki19/DocuMind-AI.git
cd DocuMind-AI
```

### 2. Install Dependencies

Ensure that Python 3.x is installed, then run the following command to install the required dependencies:

```bash
pip install -r requirements.txt
```

### 3. Set Up API Keys

Create a `.env` file in the root directory and add your **Groq API Key** (or any other required keys):

```bash
GROQ_API_KEY=<your-groq-api-key-here>
```

### 4. Run the Application

Start the Streamlit app by running:

```bash
streamlit run app.py
```

## Demo
You can try the live version of DocuMind AI here:[ DocuMind AI - Live Demo](https://multi-pdf-retrieval-augmented-generation-system.streamlit.app/)

## Usage

1. **Upload PDFs**: Use the sidebar to upload up to 10 PDF documents.
2. **Processing**: The app processes the PDFs, extracts text, and stores the embeddings in a FAISS vector store.
3. **Ask Questions**: Once the PDFs are processed, interact with the AI chatbot by asking questions based on the uploaded PDFs.

### Example Interaction

**User**: "What does the document say about climate change?"

**Assistant**: "Document 2 emphasizes the importance of renewable energy sources such as wind and solar power in reducing global carbon emissions."

## Features Breakdown

### 1. **Multiple PDF Upload and Processing**
   - Upload up to 10 PDFs and let the app process them, extracting text for easy querying.

### 2. **Efficient AI-Powered Retrieval**
   - Text from the PDFs is converted into embeddings and stored in a FAISS vector store for quick retrieval based on semantic similarity.

### 3. **Real-Time AI Responses**
   - Ask questions, and the AI generates answers using **Groq** and **Llama3** models based on the document content.

### 4. **Interactive Chat Interface**
   - The app provides an interactive chat interface where users can easily engage with the chatbot and get responses in real time.

### 5. **Upload History**
   - View and manage your upload history directly from the sidebar.

## Example Chat Interaction

**User**: "What is the conclusion of document 1?"

**Assistant**: "The conclusion of document 1 stresses the need for a global commitment to renewable energy adoption to combat climate change."

## Contribution

Feel free to open an issue or submit a pull request if you'd like to contribute to this project. All contributions are welcome!

## License

Distributed under the MIT License. See LICENSE for more details.

---

### Project by [Dhruv Yellanki](https://github.com/dhruvyellanki19)
