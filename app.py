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


text_splitter = RecursiveCharacterTextSplitter()
llm = OpenAI(model="gpt-3.5")
embeddings = OpenAIEmbeddings()



#side bar contents
with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown("""
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [Langchain](https://python.langchian.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
    - [Github](https://github.com/praj2408/Langchain-PDF-App-GUI) Repository
                
    """)
    
    st.write("Made by Prajwal Krishna.")
    
    
load_dotenv()
    
def main():
    st.header("Chat with PDF 💬")
    
    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    #st.write(pdf)
    
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, # it will divide the text into 800 chunk size each (800 tokens)
            chunk_overlap=200,
        )
        chunks = text_splitter.split_text(text=text)
        
        
        # st.write(chunks[1])
        

        knowledge_base  = FAISS.from_texts(chunks, embeddings)
            
        
        # Accept user questions/query
        query = st.text_input("Ask your questions about your PDF file")
        #st.write(query)
        
        if query:
            docs = knowledge_base.similarity_search(query)

            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)
            
            st.success(response)
            
            

    
    
    
if __name__ == "__main__":
    main()
    
