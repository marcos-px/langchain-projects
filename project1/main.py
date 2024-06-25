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

def generate_company_name_ia(SystemPrompt, UserPrompt):
    response = client.invoke_model(
        modelId=modelId,
        body=json.dumps({
            "prompt": SystemPrompt + UserPrompt,
            "maxTokens": 100,
            "stopSequences": [],
            "temperature": 0.7
            }),
        accept=accept,
        contentType=contentType,
    )
    response_body = response['body'].read()
    return json.loads(response_body)


if __name__ == "__main__":
    company_name = generate_company_name_ia(
        "Você é um assistente que responde tudo em português brasileiro", "Gere 1 função extra para advogados"
    )
    print(company_name)