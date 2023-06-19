import os
import requests
import base64
import urllib.parse

import streamlit as st
from xero_python.api_client.oauth2 import OAuth2Token
from xero_python.api_client.configuration import Configuration
from xero_python.accounting import AccountingApi

import extra_streamlit_components as stx
from datetime import datetime, timedelta

class XeroAPI:
    def __init__(self):
        self.client_id = os.getenv("API_XERO_CLIENT_ID")
        self.client_secret = os.getenv("API_XERO_API_SECRET")
        self.redirect_uri = os.getenv('API_XERO_REDIRECT_URL')
        self.scopes = os.getenv('API_XERO_SCOPE')

        self.cookies = stx.CookieManager()

    def getAuthorization(self):
        queryArgs = st.experimental_get_query_params()

        if self.cookies.get('xero_refresh_token') is None:
            if 'code' in queryArgs:
                try:
                    authorization_code = queryArgs['code'][0]
                    st.write('authorization_code', authorization_code)

                    payload = {
                        'grant_type': 'authorization_code',
                        'code': authorization_code,
                        'redirect_uri': self.redirect_uri
                    }

                    # payload = {
                    #     'grant_type': 'authorization_code',
                    #     'code': authorization_code,
                    #     'client_id': self.client_id,
                    #     'client_secret': self.client_secret,
                    #     'redirect_uri': self.redirect_uri
                    # }

                    headers = {
                        'Authorization': "Basic " + base64.b64encode((self.client_id + ":" + self.client_secret).encode()).decode(),
                        'Content-Type': "application/x-www-form-urlencoded"
                    }

                    response = requests.post('https://identity.xero.com/connect/token', headers=headers, data=payload)
                    
                    st.write(response.json())

                    if response.status_code == 200:
                        auth_response = response.json()
                        self.saveToken(auth_response['access_token'], auth_response['refresh_token'])

                    else:
                        # Handle the error case
                        st.error('Error occurred during authentication', icon=None)
                except Exception as e:
                    st.write(e.response)

            else:
                auth_url = self.getAuthorizationUrl()

                st.write(f'''
                    <a target="_self" href="{auth_url}">
                        <button>
                            Please Login to Xero
                        </button>
                    </a>
                    ''',
                    unsafe_allow_html=True
                )

    def getAuthorizationUrl(self):
        state = 'anything--123'

        # Build the authorization URL
        authorize_url = 'https://login.xero.com/identity/connect/authorize'
        params = {
            'response_type': 'code',
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri,
            'state': state,
            'scope': self.scopes,
            'prompt': 'login'
        }

        # Redirect the user to the Xero authorization URL
        # return st.redirect()
        return authorize_url + '?' + urllib.parse.urlencode(params)
    
    def saveToken(self, access_token, refresh_token):
        # for more token information:
        # https://developer.xero.com/documentation/guides/oauth2/auth-flow/#4-receive-your-tokens
        current_datetime = datetime.now()
        access_token_expiry = current_datetime + timedelta(minutes=30)
        refresh_token_expiry = current_datetime + timedelta(days=60)

        self.cookies.set('xero_access_token', access_token, expires_at=access_token_expiry)
        self.cookies.set('xero_refresh_token', refresh_token, expires_at=refresh_token_expiry)

    def getTenant(self, access_token):
        url = 'https://api.xero.com/connections'
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Accept': 'application/json'
        }

        response = requests.get(url, headers=headers)
        test_org = response.json()[2]

        self.cookies.set('xero_tenant_id', test_org['tenantId'])
    
    def refreshToken(self):
        refresh_token = self.cookies.get('refresh_token')
        payload = {
            'grant_type': 'refresh_token',
            'refresh_token': refresh_token,
            'client_id': self.client_id,
            'client_secret': self.client_secret
        }

        response = requests.post('https://identity.xero.com/connect/token', data=payload)
        st.write(response.json())

        if response.status_code == 200:
            auth_response = response.json()
            self.saveToken(auth_response['access_token'], auth_response['refresh_token'])

        else:
            # Handle the error case
            st.error('Error occurred during authentication', icon=None)

    def getBaseUrl(self):
        access_token = self.cookies.get('xero_access_token')

        if access_token is None:
            self.refreshToken()

        return {
            'url': 'https://api.xero.com/api.xro/2.0',
            'headers': {
                'Authorization': f'Bearer {access_token}',
                'Accept': 'application/json',
                'Xero-tenant-id': self.cookies.get('xero_tenant_id'),
            }
        }


    def getInvoices(self):
        req_options = self.getBaseUrl()

        response = requests.get(req_options['url'], headers=req_options['headers'])
        st.write(response.json())

    

