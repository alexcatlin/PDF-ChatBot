from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader 
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI


class PDFChatBot:
    def __init__(self):
        load_dotenv()
        self.pdf = None
        self.pdf_reader = None
        self.text = ""
        self.text_splitter = None
        self.embeddings = None
        self.knowledge_base = None
        self.query = ""
        self.llm = None
        self.chain = None

    def run(self):
        # Sets the Ui of the application
        st.set_page_config(page_title="Ask your PDF")
        st.header("PDF-CHAT Bot")
        self.pdf = st.file_uploader("Upload your pdf", type="pdf")

        # IF a pdf file is uploaded then it will be parsed into a PdfReader lib
        # I will then extract the text from the pdf and save it in the var text
        if self.pdf is not None:
            self.pdf_reader = PdfReader(self.pdf)
            self.text = ""
            for page in self.pdf_reader.pages:
                self.text += page.extract_text()

            # Split the text inside the pdf into chunks
            self.text_splitter = CharacterTextSplitter(        
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            
            self._create_embeddings()
            self._ask_query()

    # Create embeddings based on the chunks created
    def _create_embeddings(self):
        self.embeddings = OpenAIEmbeddings(openai_api_key='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')
        chunks = self.text_splitter.split_text(self.text)
        self.knowledge_base = FAISS.from_texts(chunks, self.embeddings)

    # Search the pdf for similarity and then uses qa chain lib for chatGPT's response.
    def _ask_query(self):
        self.query = st.text_input("Ask a query about your PDF: ")
        if self.query:
            docs = self.knowledge_base.similarity_search(self.query)
            self.llm = OpenAI()
            self.chain = load_qa_chain(self.llm, chain_type='stuff')
            response = self.chain.run(input_documents=docs, question=self.query)
            st.write(response)


if __name__ == '__main__':
    bot = PDFChatBot()
    bot.run()
