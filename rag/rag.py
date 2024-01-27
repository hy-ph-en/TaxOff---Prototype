from langchain.document_loaders import PyPDFLoader
import os
from pinecone import Pinecone, PodSpec
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

openai_api_key = "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"
model_name = 'text-embedding-ada-002'
pinecone_api_key = "a3ada374-cceb-4712-b907-99a33682b108"
pinecone_environment = 'us_west1-gcp'

# Docs
loader = PyPDFLoader('doc.pdf')
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_documents(documents)

# Embeddings
embeddings = OpenAIEmbeddings(
    model=model_name, 
    openai_api_key=openai_api_key
)

"""
pc = Pinecone(api_key=pinecone_api_key)


if index_name not in pc.list_indexes():
    pc.create_index(
        name='my-index',
        dimension=1536,
        metric='cosine'
    )
"""

index_name = 'langchain-chatbot'

pc = Pinecone(api_key=pinecone_api_key)
from langchain_community.vectorstores import Pinecone, 
index = Pinecone.from_documents(docs, embeddings, index_name=index_name)

# Retrives 2 most relevant contexts
retriever = index.as_retriever(search_type='similarity', search_kwargs={'k':2})

# Prompt Building
from langchain.prompts.prompt import PromptTemplate

custom_template=""" Given the following conversation and a follow up conversation, rephrase all the follow up question to be a standalone question

Chat History:
{chat_history}

Follow Up Input: {question}

Standalone Question:"""

condense_question_prompt = PromptTemplate.from_template(custom_template)


# Chatbot Chain

llm_name = 'gpt-3.5-turbo'

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

qa = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(temperature=0.1, model=llm_name, openai_api_key=openai_api_key), 
    retriever, 
    condense_question_prompt=condense_question_prompt, 
    return_source_documents=True 
)


# Running Chain

chat_history = []
query = 'What is this document about '

result = qa({"question":query, "chat_history":chat_history})

answer = result["answer"]
print(answer)

source_details = result['source_documents'][0]
print(source_details)

