import json
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

os.environ["OPENAI_API_KEY"] = "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"

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
    return_source_documents=False,
    verbose=False,
    # condense_question_prompt=qa_prompt,
)


def drag_race():
    with open("chat_history.json", "r") as f:
        chat_history = json.load(f)

    input_prompt = f"As the best tax expert, study this user's profile {json.dumps(chat_history)}, and answer the following questions:\n"

    history = []

    print("RACING")

    q1 = input_prompt + "what tax codes apply to this user?"
    r1 = pdf_qa.invoke({"question": q1, "chat_history": history})
    print(r1)
    history.append((input_prompt, r1["answer"]))

    inputs = "What deductions can this user claim if any?"
    r2 = pdf_qa.invoke({"question": inputs, "chat_history": history})
    history.append((inputs, r2["answer"]))

    inputs = "Any tax reliefs this user can claim?"
    r3 = pdf_qa.invoke({"question": inputs, "chat_history": history})
    history.append((inputs, r3["answer"]))
    inputs = "Areas where this user can save money on taxes?"
    r4 = pdf_qa.invoke({"question": inputs, "chat_history": history})
    history.append((inputs, r4["answer"]))

    print("#" * 20)
    print(history)

    with open("context.txt", "w") as f:
        for item in history:
            f.write("%s\n" % str(item))
