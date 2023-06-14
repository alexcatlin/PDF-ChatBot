from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader 
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain import HuggingFaceHub
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
        self.option = ''

    def run(self):
        # Sets the Ui of the application
        st.set_page_config(page_title="ResumePro: AI-Powered Resume Information Extraction")
        st.header("ResumePro: AI-Powered Resume Information Extraction")
        self.option = st.selectbox('What would you like to upload? ', ('Resume', 'Invoice') )
        self.llms_choice = st.selectbox('LLMS Options', ("OpenAI", "HuggingFace"))
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
        if self.option == 'Resume':
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
                'certifications':'list out the resume owner's certifications',
            
            }

            Send out the complete response and format it like a Python dictionary so that I can do eval() method on it later. Remove any bullet points
            '''
        elif self.option == 'Invoice':
            self.query = '''
            Follow this format and insert the proper information as values. Do not copy the value. if empty, just put 'none' as the value:

            {
                'slwbNo': 'the slwbno of the shipment',
                'shipper': "shipper's details (separate each details with a new line)",
                'consignee': "consignee's details (separate each details with a new line)",
                'notify_party': "notify party's details (separate each details with a new line)" ,
                'vessel':'what vessel will be used within this delivery',
                'loading_port':'port of loading',
                'discharge_port':'port of discharge',
                'packages_info': [
                    {   'mark_nos': 'total number of palletes',
                        'num_kind': ['the number of palettes for the given product'],
                        'desc': [
                            { 'cartons': 'how many cartons and what does it contain',
                                'net_weight_prod': 'net-weight of the product',
                                'temp': 'temperature of the product',
                                'ncm': 'ncm of the product'
                            }, { (add more to this list if needed, separate each products details into dictionaries inside this list) }
                            ],
                        'gross_weight': 'package gross weight',
                        'net_weight': 'package net weight'
                    }, (if necessary create more object that contains the details like the last one)
                ],

                'freight_info': 'freight, charges, etc (if no data, just leave it blank)',
                'freight_paid_at':'freight to be paid at (if no data, just leave it blank)',
                'place_date': 'place and date of issue',
                
            
            }

            Send out the complete response and format it like a Python dictionary so that I can do eval() method on it later. Remove any bullet points
            '''
    
        response = ''

        if self.query:
            docs = self.knowledge_base.similarity_search(self.query)
            if self.llms_choice == "OpenAI":
                self.llm = OpenAI(max_tokens=2048)
            elif self.llms_choice == "HuggingFace":
                repo_id = "stabilityai/stablelm-tuned-alpha-3b"
                self.llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0, "max_length":64})


            self.chain = load_qa_chain(self.llm, chain_type='stuff')
            with get_openai_callback() as cb:
                response = self.chain.run(input_documents=docs, question=self.query)
                st.write(response)
                st.write(cb)

            data = eval(response)

            if self.option == 'Resume':
                for key, value in data.items():
                    if ',' in value:
                        data[key] = [item.strip() for item in value.split(',')]
                self.name = data['name']
                self.contact = data['contact']
                self.experience = data['experience']
                self.educationalBackground = data['educationalBackground']
                self.technicalSkills = data['technicalSkills']
                self.certifications = data['certifications']

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
                st.text('Certifications: ')
                if isinstance(self.certifications, list):
                    for element in self.certifications:
                        st.text_input('Certifications: ', element, disabled=True, label_visibility='collapsed')
            if self.option == 'Invoice':
                self.slwbNo = data['slwbNo']
                self.shipper = data['shipper']
                self.consignee = data['consignee']
                self.notify_party = data['notify_party']
                self.vessel = data['vessel']
                self.loading_port = data['loading_port']
                self.discharge_port = data['discharge_port']
                self.package_info = data['packages_info']
                self.freight_info = data['freight_info']
                self.freight_paid_at = data['freight_paid_at']
                self.place_date = data['place_date']

                st.text("SLWB No.")
                st.text_input('SLWB No.: ', self.slwbNo, disabled=True, label_visibility='collapsed')
                st.text("Shipper's details: ")
                st.text_area('Shipper Details: ', self.shipper, disabled=True, label_visibility='collapsed')
                st.text("Consignee's details: ")
                st.text_area('Consignee Details: ', self.consignee, disabled=True, label_visibility='collapsed')
                st.text("Notify Party's details: ")
                st.text_area("Notify Party's details: ", self.consignee, disabled=True, label_visibility='collapsed')
                st.text("Vessel: ")
                st.text_input('Vessel: ', self.vessel, disabled=True, label_visibility='collapsed')
                st.text("Loading Port: ")
                st.text_input('Loading Port: ', self.loading_port, disabled=True, label_visibility='collapsed')
                st.text("Discharge Port: ")
                st.text_input('Discharge Port: ', self.discharge_port, disabled=True, label_visibility='collapsed')

                for info in self.package_info:
                    mark_nos = info['mark_nos']
                    num_kind = info['num_kind']
                    desc = info['desc']
                    gross_weight = info['gross_weight']
                    net_weight = info['net_weight']

                    st.text("Packages Info: ")
                    st.text('Total Palettes: ')
                    st.text_input('Mark No.: ', mark_nos, disabled=True, key=f"mark_no_{mark_nos}", label_visibility='collapsed')
                    
                    for i, num in enumerate(num_kind):
                        st.text('Number of palettes: ')
                        st.text_input('Num kind: ', num, disabled=True, key=f"num_kind_{i}", label_visibility='collapsed')
                        
                        if i < len(desc):
                            obj = desc[i]
                            cartons = obj['cartons']
                            net_weight_prod = obj['net_weight_prod']
                            temp = obj['temp']
                            ncm = obj['ncm']

                            st.text('Cartons')
                            st.text_input('Cartons: ', cartons, disabled=True, key=f"cartons_{i}", label_visibility='collapsed')
                            st.text('Net Weight')
                            st.text_input('Net Weight Prod: ', net_weight_prod, disabled=True, key=f"net_weight_prod_{i}", label_visibility='collapsed')
                            st.text('Temp:')
                            st.text_input('Temp: ', temp, disabled=True, key=f"temp_{i}", label_visibility='collapsed')
                            st.text('NCM:')
                            st.text_input('NCM: ', ncm, disabled=True, key=f"ncm_{i}", label_visibility='collapsed')
                            
                    for i in range(len(num_kind), len(desc)):
                        obj = desc[i]
                        cartons = obj['cartons']
                        net_weight_prod = obj['net_weight_prod']
                        temp = obj['temp']
                        ncm = obj['ncm']

                        st.text('Cartons')
                        st.text_input('Cartons: ', cartons, disabled=True, key=f"cartons_{i}", label_visibility='collapsed')
                        st.text('Net Weight')
                        st.text_input('Net Weight Prod: ', net_weight_prod, disabled=True, key=f"net_weight_prod_{i}", label_visibility='collapsed')
                        st.text('Temp:')
                        st.text_input('Temp: ', temp, disabled=True, key=f"temp_{i}", label_visibility='collapsed')
                        st.text('NCM:')
                        st.text_input('NCM: ', ncm, disabled=True, key=f"ncm_{i}", label_visibility='collapsed')
                    st.text('Gross Weight.')
                    st.text_input('Gross Weight: ', gross_weight, disabled=True, key="gross_weight", label_visibility='collapsed')
                    st.text('Net Weight')
                    st.text_input('Net Weight: ', net_weight, disabled=True, key="net_weight", label_visibility='collapsed')

                st.text("Freight Info: ")
                st.text_input('Freight Info: ', self.freight_info, disabled=True, label_visibility='collapsed')
                st.text("Freight Paid at: ")
                st.text_area('Freight Paid at: ', self.freight_paid_at, disabled=True, label_visibility='collapsed')
                st.text("Place & Date: ")
                st.text_input('Place & Date: ', self.place_date, disabled=True, label_visibility='collapsed')



                
if __name__ == '__main__':
    bot = PDFChatBot()
    bot.run()
