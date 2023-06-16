import os
import streamlit as st
from PyPDF2 import PdfReader 
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
import openai


class PDFChatBot:
    def __init__(self):
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
        self.response = ''


    def disable(self, b):
        st.session_state["disabled"] = b

    def run(self):
        # Sets the UI of the application
        st.set_page_config(page_title="AI PDF Reader: AI Powered PDF data Extraction")
        st.header("AI PDF Reader: AI Powered PDF data Extraction")
        self.option = st.selectbox('What would you like to upload?', ('Resume', 'Bill of loading', "Ask your pdf", "Procurement"))
        self.model = st.selectbox('Model Options', ("text-davinci-003", "gpt-3.5-turbo"))
        self.pdf = st.file_uploader("Upload a pdf", type="pdf")

        # If a pdf file is uploaded, it will be parsed into a PdfReader lib
        # The text will be extracted from the pdf and saved in the 'text' variable
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

            if self.option == "Ask your pdf":
                self._create_embeddings()
                self._ask_query()
                st.write(self.response)
            else:
                placeholder = st.empty()
                self.button = placeholder.button('Extract data', key="button", disabled=False)

                if st.session_state.get("button", False):
                    st.session_state.disabled = False
                elif st.session_state.get("button", False):
                    st.session_state.disabled = True

                if self.button:
                    placeholder.button('Extract data', disabled=True, key='2')
                    loading_text = st.text("Loading the data please wait...")
                    self._create_embeddings()
                    self._ask_query()
                    loading_text.empty()

    # Create embeddings based on the chunks created
    def _create_embeddings(self):
        self.embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        chunks = self.text_splitter.split_text(self.text)
        self.knowledge_base = FAISS.from_texts(chunks, self.embeddings)

    # Search the pdf for similarity and then use qa chain lib for chatGPT's response.
    def _ask_query(self):
        if self.option == 'Resume':
            self.query = '''
            Follow this format and insert the owner's information in the values. Do not copy the value:

            {   
                'docType': 'Answer with True or False, is this a resume?'
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
                'educationalBackground': { 
                    'school': 'the school attended', 
                    'course': 'the course taken', 
                    'year': 'year started and ended' 
                },
                'technicalSkills': 'only give 5 of the most notable technical skills that can apply to full stack development',
                'certifications': 'list out the resume owner\'s certifications',
            }

            Send out the complete response and format it like a Python dictionary so that I can do eval() method on it later. Remove any bullet points
            '''
        elif self.option == 'Bill of loading':        
            self.query = '''
            Follow this format and insert the proper information as values. Do not copy the value. If empty, just put 'none' as the value:

            {   'docType': 'Answer with True or False, is this doc a bill of loading?',
                'slwbNo': 'the slwbno of the shipment',
                'shipper': "shipper's details (separate each detail with a new line)",
                'consignee': "consignee's details (separate each detail with a new line)",
                'notify_party': "notify party's details (separate each detail with a new line)",
                'vessel': 'what vessel will be used within this delivery',
                'loading_port': 'port of loading',
                'discharge_port': 'port of discharge',
                'packages_info': [
                    {
                        'mark_nos': 'total number of palletes',
                        'num_kind': ['the number of palettes for the given product'],
                        'desc': [
                            {
                                'cartons': 'how many cartons and what does it contain',
                                'net_weight_prod': 'net-weight of the product',
                                'temp': 'temperature of the product',
                                'ncm': 'ncm of the product'
                            },
                            # (add more to this list if needed, separate each product's details into dictionaries inside this list)
                        ],
                        'gross_weight': 'package gross weight',
                        'net_weight': 'package net weight'
                    },
                    # (if necessary create more object that contains the details like the last one)
                ],
                'freight_info': 'freight, charges, etc (if no data, just leave it blank)',
                'freight_paid_at': 'freight to be paid at (if no data, just leave it blank)',
                'place_date': 'place and date of issue',
            }

            Send out the complete response and format it like a Python dictionary so that I can do eval() method on it later. Remove any bullet points
            '''
        elif self.option == 'Ask your pdf':
            self.query = st.text_input("Ask your pdf?", key="ask_pdf_input")
        elif self.option == "Procurement":
            self.query = '''
                Follow this format and insert the proper information as values. Do not copy the value. If empty, just put 'none' as the value:

                {
                    'docType' 'Answer with True or False, is this doc a procurement doc based on the doc template? if the doc is a resume or a bill of loading answer False',
                    'customer_name': 'the customer name stated in the paper.',
                    'quote_info': {
                        'quote_address': 'The address of the provider of the quote (add newline to every new bit of information.)',
                        'quote_number': 'Quote Number or ID',
                        'quote_creation_date': 'The Quote Creation Date',
                        'quote_expiration_date': 'The expiration date of the quote',
                    },
                    'customer_details': {
                        'customer_number': 'The customer number id',
                        'payment_method': 'The payment method used for this quote',
                        'customer_information': 'The customer\'s information'                   
                    },
                    'billing': {
                        'sales_rep': 'The details of the sales representative (add newline to every new bit of information.)',
                        'bill_to': 'the information the \'bill to\' contains',
                        'mail_to': 'the information the \'mail to\' contains',
                        'ship_to': 'the information the \'ship to\' contains',
                    },
                    'pricing_summary': [
                        {
                            'product name': 'the product name or description',
                            'qty': 'the product quantity',
                            'list_price': 'product\'s list price',
                            'unit_price': 'product\'s unit price',
                            'net_price': 'product\'s net price',
                            'mark_up': 'The markup in price of list price and unit price in percentage add a & sign at the end'
                        }
                    ]
                }
            '''
        
        if self.query:
            docs = self.knowledge_base.similarity_search(self.query)
            self.llm = ChatOpenAI(max_tokens=2048, model_name=self.model)
            self.chain = load_qa_chain(self.llm, chain_type='stuff')
            try:
                self.response = self.chain.run(input_documents=docs, question=self.query)
                st.write(self.response)
                if (self.option != 'Ask your pdf'):
                    self.data = eval(self.response)

                if self.option == 'Resume':
                    if self.data['docType'] == 'True':
                        self.resume_query()
                    else:
                        st.markdown(""":red[Error: This file isn't a resume file. Please upload a resume file or choose other options] """)
                        
                    
                elif self.option == 'Bill of loading':
                    if self.data['docType'] == 'True':
                        self.bill_of_loading()
                    else:
                        st.markdown(""":red[Error: This file isn't a bill of loading file. Please upload a bill of loading file or choose other options] """)
    
                elif self.option == 'Procurement':
                    if self.data['docType'] == 'True':
                        self.procurement()
                    else:
                        st.markdown(""":red[Error: This file isn't a procurement file. Please upload a procurement file or choose other options] """)
                
            except openai.error.InvalidRequestError:
                st.markdown(""":red[Error: Maximum context length exceeded. Please cut down your pdf and upload only the necessary pages.] """)
                return

    # Logic behind the different types of UI per query
    def resume_query(self):
        for key, value in self.data.items():
            if ',' in value:
                self.data[key] = [item.strip() for item in value.split(',')]
        name = self.data['name']
        contact = self.data['contact']
        experiences = self.data['experience']
        educationalBackground = self.data['educationalBackground']
        technicalSkills = self.data['technicalSkills']
        certifications = self.data['certifications']

        st.text('Name: ')
        st.text_input('Name: ', name, disabled=True, label_visibility='collapsed')
        st.text('Contact Information: ')
        if isinstance(contact, list):
            for element in contact:
                st.text_input('Contacts: ', element, disabled=True, label_visibility='collapsed')
        else:
            st.text_input('Contact: ', contact, disabled=True, label_visibility='collapsed')

        st.text('Experiences: ')
        for experience in experiences:
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

        school = educationalBackground['school']
        course = educationalBackground['course']
        year = educationalBackground['year']
        st.text('Education: ')
        st.text('Institution')
        st.text_input('School Attended: ', school, disabled=True, label_visibility='collapsed')
        st.text('Course')
        st.text_input('School Attended: ', course, disabled=True, label_visibility='collapsed')
        st.text('Year Graduated')
        st.text_input('School Attended: ', year, disabled=True, label_visibility='collapsed')

        st.text('Technical Skills: ')
        if isinstance(technicalSkills, list):
            for element in technicalSkills:
                st.text_input('Technical Skills: ', element, disabled=True, label_visibility='collapsed')
        st.text('Certifications: ')
        if isinstance(certifications, list):
            for element in certifications:
                st.text_input('Certifications: ', element, disabled=True, label_visibility='collapsed')

    def bill_of_loading(self):
        slwbNo = self.data['slwbNo']
        shipper = self.data['shipper']
        consignee = self.data['consignee']
        notify_party = self.data['notify_party']
        vessel = self.data['vessel']
        loading_port = self.data['loading_port']
        discharge_port = self.data['discharge_port']
        package_info = self.data['packages_info']
        freight_info = self.data['freight_info']
        freight_paid_at = self.data['freight_paid_at']
        place_date = self.data['place_date']

        st.text("SLWB No.")
        st.text_input('SLWB No.: ', slwbNo, disabled=True, label_visibility='collapsed')
        st.text("Shipper's details: ")
        st.text_area('Shipper Details: ', shipper, disabled=True, label_visibility='collapsed')
        st.text("Consignee's details: ")
        st.text_area('Consignee Details: ', consignee, disabled=True, label_visibility='collapsed')
        st.text("Notify Party's details: ")
        st.text_area("Notify Party's details: ", notify_party, disabled=True, label_visibility='collapsed')
        st.text("Vessel: ")
        st.text_input('Vessel: ', vessel, disabled=True, label_visibility='collapsed')
        st.text("Loading Port: ")
        st.text_input('Loading Port: ', loading_port, disabled=True, label_visibility='collapsed')
        st.text("Discharge Port: ")
        st.text_input('Discharge Port: ', discharge_port, disabled=True, label_visibility='collapsed')

        for info in package_info:
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
        st.text_input('Freight Info: ', freight_info, disabled=True, label_visibility='collapsed')
        st.text("Freight Paid at: ")
        st.text_area('Freight Paid at: ', freight_paid_at, disabled=True, label_visibility='collapsed')
        st.text("Place & Date: ")
        st.text_input('Place & Date: ', place_date, disabled=True, label_visibility='collapsed')

    def procurement(self):
        customer_name = self.data['customer_name']
        quote_info = self.data['quote_info']
        customer_details = self.data['customer_details']
        billing = self.data['billing']
        pricing_summary = self.data['pricing_summary']

        st.header("Procurement")
        st.text('Customer Name: ')
        st.text_input('Customer Name: ', customer_name, disabled=True, label_visibility='collapsed')
        st.text('Quote Informations: ')
        for key, value in quote_info.items():
            if key == 'quote_address':
                st.text("Quote Address:")
                st.text_area('', value, disabled=True, key="quote_address", label_visibility='collapsed')
            else:
                st.text(key.capitalize() + ':')
                st.text_input('', value, disabled=True, key=f"quote_info_{key}", label_visibility='collapsed')

        st.text("Customer Information")
        for key, value in customer_details.items():
            st.text(key.capitalize() + ':')
            st.text_input('', value, disabled=True, key=f"customer_details_{key}", label_visibility='collapsed')                 

        for key, value in billing.items():
            st.text(key.capitalize() + ':')
            st.text_area('', value, disabled=True, key=f"billing_{key}", label_visibility='collapsed')

        st.text('Pricing Summary:')
        for i, product in enumerate(pricing_summary):
            st.text('Product Name:')
            st.text_input('', product['product name'], disabled=True, key=f"product_name_{i}", label_visibility='collapsed')
            st.text('Quantity:')
            st.text_input('', str(product['qty']), disabled=True, key=f"quantity_{i}", label_visibility='collapsed')
            st.text('List Price:')
            st.text_input('', str(product['list_price']), disabled=True, key=f"list_price_{i}", label_visibility='collapsed')
            st.text('Unit Price:')
            st.text_input('', str(product['unit_price']), disabled=True, key=f"unit_price_{i}", label_visibility='collapsed')
            st.text('Net Price:')
            st.text_input('', str(product['net_price']), disabled=True, key=f"net_price_{i}", label_visibility='collapsed')
            st.text('Markup:')
            st.text_input('', product['mark_up'], disabled=True, key=f"markup_{i}", label_visibility='collapsed')

if __name__ == '__main__':
    bot = PDFChatBot()
    bot.run()
