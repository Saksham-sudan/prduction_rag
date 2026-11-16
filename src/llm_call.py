from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from src.retriver import retriver_func
import time

llm = ChatGoogleGenerativeAI(
    model = 'gemini-3.1-flash-lite'
)

def call_func(input_prompt):
    print("CALLING.....")
    messages = [
        SystemMessage(content="you are an helpful QA bot which makes use of the retrived relevant context regarding the user input, refuse to anser if no relevant context present")
        ]
    messages.append(HumanMessage(content = input_prompt))
    llm_tool = llm.bind_tools([retriver_func])
    tool_response = llm_tool.invoke(messages)
    time.sleep(4)

    if not tool_response.tool_calls:
        messages.append(tool_response)
        return tool_response, []
    else:
        messages.append(tool_response)
        tool_msg = retriver_func.invoke(tool_response.tool_calls[0])
        time.sleep(4)
        messages.append(tool_msg)
        final_response = llm.invoke(messages)
        time.sleep(4)
        messages.append(final_response)
        return final_response, tool_msg.artifact