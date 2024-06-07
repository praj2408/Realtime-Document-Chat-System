import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks import get_openai_callback # helps us to know how much it costs us for each query
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain



from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain

from dotenv import load_dotenv

load_dotenv()

text_splitter = RecursiveCharacterTextSplitter()
llm = OpenAI(model="gpt-4")
embeddings = OpenAIEmbeddings()



    
def main():
    # Page configuration for better aesthetics
    st.set_page_config(page_title="PDF Summarizer", layout="centered")

    # Custom styles for a better-looking design with a centered logo and white background
    st.markdown("""
    <style>
    /* Overall body style */
    body {
        background-color: white;  /* Ensuring the background is white */
        margin: 0;
        font-family: 'Helvetica Neue', sans-serif;
    }
    /* Streamlit's automatic styling adjustments */
    .css-18e3th9 {
        padding: 0;
    }
    /* Centering the logo with adjustments for mobile responsiveness */
    .logo-img {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 150px;  /* Adjust the width as necessary */
        margin-top: 10px;
        margin-bottom: 20px;
    }
    /* Streamlit components and widgets styling */
    .stTextInput, .stButton, .stAlert {
        margin: 10px 0;
    }
    .stTextInput input {
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #ccc;
    }
    .stButton > button {
        width: 100%;
        color: white;
        background-color: #008CBA;
    }
    .stAlert {
        border-radius: 5px;
        background-color: #dbf0f7;
    }
    </style>
    <img src="https://static.wixstatic.com/media/55efa7_41b05c38b58649d6bc98c14aa277a767~mv2.png/v1/fill/w_452,h_120,al_c,q_85,usm_0.66_1.00_0.01,enc_auto/D2K%20Technologies_CMYK_PNG%20(1).png" class="logo-img">
    """, unsafe_allow_html=True)

    st.title("PDF Summarizer")

    # File uploader for PDF
    pdf = st.file_uploader("Upload your PDF", type=["pdf"], help="Select a PDF file to upload.")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text=text)

        knowledge_base = FAISS.from_texts(chunks, embedding=OpenAIEmbeddings())

        query = st.text_input("Ask your questions about the PDF file", placeholder="Type your question here...")

        if query:
            docs = knowledge_base.similarity_search(query)
            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type='stuff')
            response = chain.run(input_documents=docs, question=query)

            st.success(response)

if __name__ == "__main__":
    main()