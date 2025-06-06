import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()

# ページ設定
st.set_page_config(page_title="LLM Chat", page_icon="💬")

# タイトル
st.title("💬 LLM Chat")

# チャット履歴の初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

# チャット履歴の表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ユーザー入力
if prompt := st.chat_input("メッセージを入力してください"):
    # ユーザーメッセージの追加
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AIの応答
    with st.chat_message("assistant"):
        chat = ChatOpenAI(
            model_name="gpt-4o",
            temperature=0.7,
        )
        messages = [
            SystemMessage(content="あなたは親切なアシスタントです。"),
            HumanMessage(content=prompt),
        ]
        response = chat.invoke(messages)
        st.markdown(response.content)
        st.session_state.messages.append(
            {"role": "assistant", "content": response.content}
        )
