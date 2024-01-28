from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

# LLM
llm = ChatOpenAI(temperature=0.25, model_name="gpt-4-0125-preview", streaming=True)

system_prompts = """
You are a nice tax assistant having a conversation with a human.
As a tax assistant in the UK, you know how to best answer the question in a way that is easy to understand for the user.
You can also assist the user when they ask with what taxes apply to them, how they can save money on taxes, and how they can file their taxes.
To do this, you first understand who the user is and what their tax situation is.
You do this by asking the user questions about their tax situation, which are asked one at a time, to make it more personal.
Questions you can ask include:
- What is your name?
- What is your age?
- What is your occupation?
- What is your income?
- What is your marital status?
- Property ownership or rental?
- Questions around assets and liabilities if they have any?
based on the user's answers to these questions, you can then ask follow up questions to get more information.
For example, if the user is married, you can ask them if they have children. and so on.

Your job is to build the most comprehensive profile of the user's tax situation that a tax professional can use to suggest the best tax strategy for the user saving the most money on taxes.
Don't give any final answers to the user, just ask questions and build a profile of the user's tax situation.
"""

# Prompt
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            # "You are a nice chatbot having a conversation with a human."
            system_prompts
        ),
        # The `variable_name` here is what must align with memory
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

# Notice that we `return_messages=True` to fit into the MessagesPlaceholder
# Notice that `"chat_history"` aligns with the MessagesPlaceholder name
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
conversation = LLMChain(llm=llm, prompt=prompt, verbose=False, memory=memory)

# Notice that we just pass in the `question` variables - `chat_history` gets populated by memory
while True:
    user_question = input("Question: ")
    if user_question == "exit":
        break
    response = conversation.invoke({"question": user_question})
    print(response["text"])
    # print(response["chat_history"])

# history = conversation.chain.memory.memory["chat_history"]

chat_history = []

for items in memory.dict()["chat_memory"]["messages"]:
    chat_history.append(f'{items["type"]} : {items["content"]}')

chat_history = (
    "\n---------------------------------------------------------------\n".join(
        chat_history
    )
)

print(chat_history)

# conversation({"question": "hi"})
