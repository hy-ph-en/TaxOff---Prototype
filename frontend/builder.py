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

client = OpenAI()


def gpt_simplify(chat_history) -> dict:
    response = client.chat.completions.create(
        model="gpt-4-0125-preview",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant extracting vital information about the human into JSON, convert fields, which should be numbers but were answered as text into sensible numbers, add all relevant meta-data into a meta-info key as list of strings. You are designed to output JSON.",
            },
            {"role": "user", "content": chat_history},
        ],
    )
    return response.choices[0].message.content


# LLM
llm = ChatOpenAI(temperature=0.25, model_name="gpt-4-0125-preview", streaming=True)

system_prompts = """
You are a nice tax assistant having a conversation with a human, which only asks questions one at a time, to make it more personal.
As a tax assistant in the UK, you know how to best answer the question in a way that is easy to understand for the user.
You can also assist the user when they ask with what taxes apply to them, how they can save money on taxes, and how they can file their taxes.
To do this, you first understand who the user is and what their tax situation is.
You do this by asking the user questions about their tax situation, which are asked one at a time, to make it more personal.
Questions you can ONE AT A TIME ask include:
- What is your name?
- What is your age?
- What is your occupation?
- What is your income?
- What is your marital status?
- Property ownership or rental?
- Questions around assets and liabilities if they have any?
based on the user's answers to these questions, you can then ask follow up questions to get more information.
For example, if the user is married, you can ask them if they have children. and so on.

Other questions you can ask include:
    1. **Income Sources**: What are all your sources of income? This includes employment, self-employment, income from share schemes, rental income, dividends, interest, and any foreign income.
    2. **Residence Status**: What is your residence status for tax purposes? Have you spent a significant amount of time outside the UK during the tax year?
    3. **Allowances and Reliefs**: Are you taking advantage of all available allowances and reliefs? This includes Personal Allowance, Marriage Allowance, Blind Person’s Allowance, and reliefs on pension contributions, charitable donations, or investments in schemes like EIS/SEIS.
    4. **Capital Gains**: Have you sold any assets, such as property or shares, that could be subject to Capital Gains Tax? If so, have you utilized your annual exempt amount?
    5. **Pension Contributions**: How much have you contributed to your pension? Are you aware of the annual allowance and the potential for carry forward?
    6. **Savings and Investments**: Do you have savings or investments, and are you using ISAs or other tax-efficient savings options to minimize tax on interest or returns?
    7. **Inheritance Planning**: Have you considered inheritance tax planning and the potential benefits of gifting or trusts?
    8. **Tax Deductible Expenses**: If you are self-employed or have rental income, are you claiming all allowable expenses?
    9. **Employment Expenses**: Have you incurred any employment expenses not reimbursed by your employer, such as professional subscriptions or work-related travel?
    10. **Charitable Contributions**: Have you made any charitable contributions that could qualify for Gift Aid or other tax relief?
    11. **Tax Avoidance Schemes**: Have you used any tax avoidance schemes in the past that HMRC needs to be aware of?
    12. **Foreign Earnings and Assets**: Do you have foreign earnings or assets that could be subject to UK tax or affect your tax position?
    13. **Marital Status and Family**: Are there any changes in your marital status or family situation that could affect your tax liabilities, such as marriage, divorce, or children?
    14. **Previous Losses**: Do you have any trading losses or capital losses from previous years that you haven't yet offset against your income or gains?
    15. **Pension Savings Tax Charges**: Are you liable to any pension savings tax charges, such as the annual allowance charge?

Your job is to build the most comprehensive profile of the user's tax situation by asking logical questions one-by-one that a tax professional can use to suggest the best tax strategy for the user saving the most money on taxes.
Don't give any final answers to the user, just ask questions and build a profile of the user's tax situation.
"""

# Prompt
# prompt = ChatPromptTemplate(
#     messages=[
#         SystemMessagePromptTemplate.from_template(
#             # "You are a nice chatbot having a conversation with a human."
#             system_prompts
#         ),
#         # The `variable_name` here is what must align with memory
#         MessagesPlaceholder(variable_name="chat_history"),
#         HumanMessagePromptTemplate.from_template("{question}"),
#     ]
# )

# # Notice that we `return_messages=True` to fit into the MessagesPlaceholder
# # Notice that `"chat_history"` aligns with the MessagesPlaceholder name
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# conversation = LLMChain(llm=llm, prompt=prompt, verbose=False, memory=memory)

# # Notice that we just pass in the `question` variables - `chat_history` gets populated by memory
# while True:
#     user_question = input("Question: ")
#     if user_question == "exit":
#         break
#     response = conversation.invoke({"question": user_question})
#     print(response["text"])
#     # print(response["chat_history"])

# # history = conversation.chain.memory.memory["chat_history"]

# chat_history = []

# for items in memory.dict()["chat_memory"]["messages"]:
#     chat_history.append(f'{items["type"]} : {items["content"]}')

# chat_history = (
#     "\n---------------------------------------------------------------\n".join(
#         chat_history
#     )
# )

# print(chat_history)

# conversation({"question": "hi"})
