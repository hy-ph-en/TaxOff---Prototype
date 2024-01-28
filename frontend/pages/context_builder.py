import streamlit as st
import utils
import json
from builder import system_prompts, gpt_simplify
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI
from openai import OpenAI
from langchain.memory import ConversationBufferMemory
from streaming import StreamHandler

# Assuming utils and other necessary imports are correctly defined

st.set_page_config(page_title="Chatbot", page_icon="💬")
st.header("Basic Chatbot")
st.write("Allows users to interact with the LLM")

client = OpenAI()

llm = ChatOpenAI(temperature=0.25, model_name="gpt-4-0125-preview", streaming=True)
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
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chain = LLMChain(llm=llm, prompt=prompt, verbose=False, memory=memory)
# return conversation, memory


def build_history(memory):
    chat_history = []

    for items in memory.dict()["chat_memory"]["messages"]:
        chat_history.append(f'{items["type"]} : {items["content"]}')

    chat_history = (
        "\n---------------------------------------------------------------\n".join(
            chat_history
        )
    )
    return chat_history


class Chatbot:
    def __init__(self):
        utils.configure_openai_api_key()
        self.openai_model = "gpt-3.5-turbo"
        self.custom_responses = [
            "This is the first custom response.",
            "Here's another response, tailored to your second question.",
            "Continuing our conversation, here's the third response.",
            # Add more responses as needed
        ]
        # Initialize response_index in session_state if it doesn't exist
        if "response_index" not in st.session_state:
            st.session_state.response_index = 0

    @st.cache_resource
    def setup_chain(_self):
        # llm = OpenAI(model_name=self.openai_model, temperature=0, streaming=True)
        llm = ChatOpenAI(
            temperature=0.25, model_name="gpt-4-0125-preview", streaming=True
        )
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
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        conversation = LLMChain(llm=llm, prompt=prompt, verbose=False, memory=memory)
        return conversation

    @utils.enable_chat_history_profile
    def context_builder(self):
        chain = self.setup_chain()
        user_query = st.chat_input(placeholder="Type your question here!")
        if user_query:
            utils.display_msg(user_query, "user")
            with st.chat_message("assistant"):
                if (
                    user_query == "exit"
                    or user_query == "quit"
                    or user_query == "bye"
                    or user_query == "thanks"
                ):
                    history = build_history(chain.memory)
                    simple_json = gpt_simplify(history)
                    st.write(simple_json)

                # st_cb = StreamHandler(st.empty())
                inputs = {
                    "question": user_query,
                }
                response = chain.invoke(inputs)
                response = response["text"]
                print(response)
                print(chain.memory.dict())
                # if st.session_state.response_index < len(self.custom_responses):
                #     # Get the next custom response from the list using session_state
                #     custom_response = self.custom_responses[
                #         st.session_state.response_index
                #     ]
                #     st.session_state.response_index += 1
                # else:
                #     # Once all responses are used up, return "Thank you"
                #     custom_response = "Thank you"
                #     self.save_messages_to_file(st.session_state.messages)
                #     # Reset the index if you want to start over, or disable the input if not
                #     # st.session_state.response_index = 0
                #     # To disable the input, you could use something like st.stop()

                # Display the custom response
                st.write(response)

                # Add the custom response to the session state messages
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )

    def save_messages_to_file(self, messages):
        # Save the messages to a file in JSON format
        with open("chat_history.json", "w") as f:
            json.dump(messages, f, indent=2)


if __name__ == "__main__":
    obj = Chatbot()
    obj.context_builder()
