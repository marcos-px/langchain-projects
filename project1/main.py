from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage
from boto3 import session as boto3_session
from dotenv import load_dotenv
import logging
import os
import json

load_dotenv()

logger = logging.getLogger(__name__)
profile_name = os.getenv("AWS_PROFILE")
region_name = os.getenv("AWS_REGION")

session = boto3_session.Session(
    profile_name=profile_name, 
    region_name=region_name
    )

client = session.client('bedrock-runtime')

modelId = os.getenv("MODEL_ID")
accept = 'application/json'
contentType = 'application/json'

def generate_company_name_ia(system_prompt, user_prompt):
    
    system_message = SystemMessage(content=system_prompt)
    user_message = HumanMessage(content=user_prompt)
    
    final_prompt = f"{system_message.content}\n{user_message.content}"
    
    response = client.invoke_model(
        modelId=modelId,
        body=json.dumps({
            "prompt": final_prompt,
            "maxTokens": 100,
            "stopSequences": [],
            "temperature": 0.7
        }),
        accept=accept,
        contentType=contentType
    )
    
    response_body = response['body'].read()
    return json.loads(response_body)

if __name__ == "__main__":
    system_prompt = "Você é um assistente IA que sempre responde em Português do Brasil."
    user_prompt = "Diga uma função extra para um advogado Júnior numa empresa."

    company_name_response = generate_company_name_ia(system_prompt, user_prompt)
    print(company_name_response)