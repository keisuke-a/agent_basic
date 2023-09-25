import openai
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st

st_callback = StreamlitCallbackHandler(st.container())

openai.api_key = st.secrets.OpenAIAPI.openai_api_key
llm = ChatOpenAI(temperature=0, streaming=True)
tools = load_tools(["ddg-search"]) # DuckDuckGoの検索ツールをロードする

# ReActエージェントを初期化する。このエージェントはzero-shot学習と説明を組み合わせたもの
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

if prompt := st.chat_input(): # Streamlit のチャット入力がある場合に実行する
    st.chat_message("user").write(prompt) # ユーザの入力メッセージをStreamlitのチャットに表示する
    with st.chat_message("assistant"): # アシスタントの応答を表示するためのブロックを開始する
        st_callback = StreamlitCallbackHandler(st.container()) # Streamlitのコンテナをコールバックとして使用するハンドラを初期化する****
        response = agent.run(prompt, callbacks=[st_callback]) # エージェントを使って
        st.write(response) # 応答をStreamlitのチャットに表示する
