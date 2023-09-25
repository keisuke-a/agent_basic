from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st
from langchain.tools import DuckDuckGoSearchRun
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.callbacks import StreamlitCallbackHandler
from langchain import LLMMathChain

openai.api_key = st.secrets.OpenAIAPI.openai_api_key
st_callback = StreamlitCallbackHandler(st.container())
search = DuckDuckGoSearchRun()
llm = ChatOpenAI(temperature=0, streaming=True, model="gpt-3.5-turbo")
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

# 使用可能なツールのリストを作成する
tools = [
    Tool(
        name = "ddg-search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions"
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math"
    ),
]

# エージェントを初期化する。このエージェントはOpenAIの関数を使用
agent = initialize_agent(
    tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True
)

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(prompt, callbacks=[st_callback])
        st.write(response)
