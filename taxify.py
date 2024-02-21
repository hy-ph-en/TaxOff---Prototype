import os

from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import ChatOpenAI

from langchain_openai import OpenAIEmbeddings

# from langchain.embeddings.openai import OpenAIEmbeddings

# SET OPENAI_API_KEY
if "OPENAI_API_KEY" not in os.environ:
    print(
        "WARNING: OPENAI_API_KEY not set, either run: export OPEN_API_KEY=<key> or set it in the code in taxify.py and rag_step.py"
    )
    os.environ["OPENAI_API_KEY"] = ""

documents = []
# Create a List of Documents from all of our files in the ./docs folder
for file in os.listdir("./"):
    if file.endswith(".pdf"):
        pdf_path = "./docs/" + file
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
    elif file.endswith(".docx") or file.endswith(".doc"):
        doc_path = "./docs/" + file
        loader = Docx2txtLoader(doc_path)
        documents.extend(loader.load())
    elif file.endswith(".txt"):
        text_path = "./" + file
        loader = TextLoader(text_path)
        documents.extend(loader.load())

# Split the documents into smaller chunks

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=128)
documents = text_splitter.split_documents(documents)

print(f"Number of documents: {len(documents)}")
# print(f"Number of chunks: {len(documents[0])}")
# print("First chunk of first document:", documents[-1])


# Convert the document chunks to embedding and save them to the vector store
emb_func = OpenAIEmbeddings(
    # model="text-embedding-3-large",
    deployment="text-embedding-3-large",
    chunk_size=256,
    # show_progress_bar=True,
    retry_max_seconds=120,
)

if os.path.exists("./data"):
    print("Loading from disk...")
    vectordb = Chroma(persist_directory="./data", embedding_function=emb_func)
    vectordb.get()
else:
    vectordb = Chroma.from_documents(
        documents,
        embedding=OpenAIEmbeddings(
            # model="text-embedding-3-large",
            deployment="text-embedding-3-large",
            chunk_size=256,
            # show_progress_bar=True,
            retry_max_seconds=120,
        ),
        persist_directory="./data",
    )

    vectordb.persist()

# create our Q&A chain

pdf_qa = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(temperature=0.25, model_name="gpt-4-0125-preview", streaming=True),
    retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    verbose=False,
    # condense_question_prompt=qa_prompt,
)

yellow = "\033[0;33m"
green = "\033[0;32m"
white = "\033[0;39m"

chat_history = []
# print(
#     f"{yellow}---------------------------------------------------------------------------------"
# )
# print(
#     "Welcome to the DocBot. You are now ready to start interacting with your documents"
# )
# print(
#     "---------------------------------------------------------------------------------"
# )
while True:
    query = input(f"{green}Prompt: ")
    if query == "exit" or query == "quit" or query == "q" or query == "f":
        print("Exiting")
        exit()
    if query == "":
        continue
    result = pdf_qa.invoke({"question": query, "chat_history": chat_history})
    print(f"{white}Answer: " + result["answer"])

    chat_history.append((query, result["answer"]))  # The LangChain repository
