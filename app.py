from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader 
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback



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
        st.set_page_config(page_title="ResumePro: AI-Powered Resume Information Extraction")
        st.header("ResumePro: AI-Powered Resume Information Extraction")
        self.pdf = st.file_uploader("Upload a resume", type="pdf")

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
                chunk_size=1250,
                chunk_overlap=200,
                length_function=len
            )
            
            self._create_embeddings()
            self._ask_query()

    # Create embeddings based on the chunks created
    def _create_embeddings(self):
        self.embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        chunks = self.text_splitter.split_text(self.text)
        self.knowledge_base = FAISS.from_texts(chunks, self.embeddings)

    # Search the pdf for similarity and then uses qa chain lib for chatGPT's response.
    def _ask_query(self):
        self.query = '''
            Follow this format and insert the owner's information in the values. Do not copy the value:

            {
                'name': "owner's name",
                'contact': "owner's contact information separated by commas",
                'experience': [
                    {
                        'company_name': 'first work experience company name',
                        'job_date': 'from date and to date of the job',
                        'job_title': 'job title for this experience',
                        'job_description': 'summarize the experience description'
                    },
                    # Add more experiences if relevant
                ],
                'educationalBackground':
                    { 
                        'school': 'the school attended', 
                        'course': 'the course taken', 
                        'year': 'year started and ended' 
                    },
                'technicalSkills': 'only give 5 of the most notable technical skills that can apply to full stack development',
            
            }

            Send out the complete response and format it like a Python dictionary so that I can do eval() method on it later. Remove any bullet points
            '''
        
        response = ''

        if self.query:
            docs = self.knowledge_base.similarity_search(self.query)
            self.llm = OpenAI(max_tokens=512)
            self.chain = load_qa_chain(self.llm, chain_type='stuff')
            response = self.chain.run(input_documents=docs, question=self.query)
            st.write(response)
            st.write(self.llm)
            with get_openai_callback() as cb:
                response = self.chain.run(input_documents=docs, question=self.query)
                print(cb)
                st.write(cb)
                data = eval(response)
                for key, value in data.items():
                    if ',' in value:
                        data[key] = [item.strip() for item in value.split(',')]

                self.name = data['name']
                self.contact = data['contact']
                self.experience = data['experience']
                self.educationalBackground = data['educationalBackground']
                self.technicalSkills = data['technicalSkills']
                # self.certifications = data['certifications']

                st.text('Name: ')
                st.text_input('Name: ', self.name, disabled=True, label_visibility='collapsed')
                st.text('Contact Information: ')
                if isinstance(self.contact, list):
                    for element in self.contact:
                        st.text_input('Contacts: ', element, disabled=True, label_visibility='collapsed')
                else:
                    st.text_input('Contact: ', self.contact, disabled=True, label_visibility='collapsed')

                st.text('Experiences: ')
                for experience in self.experience:
                    company_name = experience['company_name']
                    job_date = experience['job_date']
                    job_title = experience['job_title']
                    job_description = experience['job_description']

                    st.text('Company Name')
                    st.text_input('Company Name: ', company_name, disabled=True, label_visibility='collapsed')
                    st.text('Job Date')
                    st.text_input('Job Date: ', job_date, disabled=True, label_visibility='collapsed')
                    st.text('Job Title')
                    st.text_input('Job Title: ', job_title, disabled=True, label_visibility='collapsed')
                    st.text('Job Description')
                    st.text_area('Job Description: ', value=job_description, disabled=True, label_visibility='collapsed')

                self.school = self.educationalBackground['school']
                self.course = self.educationalBackground['course']
                self.year = self.educationalBackground['year']
                st.text('Education: ')
                st.text('Institution')
                st.text_input('School Attended: ', self.school, disabled=True, label_visibility='collapsed')
                st.text('Course')
                st.text_input('School Attended: ', self.course, disabled=True, label_visibility='collapsed')
                st.text('Year Graduated')
                st.text_input('School Attended: ', self.year, disabled=True, label_visibility='collapsed')


                # st.text_input('Education: ', self.educationalBackground, disabled=True, label_visibility='collapsed')
                st.text('Technical Skills: ')
                if isinstance(self.technicalSkills, list):
                    for element in self.technicalSkills:
                        st.text_input('Technical Skills: ', element, disabled=True, label_visibility='collapsed')
                # st.text('Certifications: ')
                # if isinstance(self.certifications, list):
                #     for element in self.certifications:
                #         st.text_input('Certifications: ', element, disabled=True, label_visibility='collapsed')

if __name__ == '__main__':
    bot = PDFChatBot()
    bot.run()
