from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()


def main():
    # LLMの初期化
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

    # プロンプトテンプレートの作成
    prompt = PromptTemplate(
        input_variables=["topic"],
        template="以下のトピックについて3つの重要なポイントを日本語で説明してください: {topic}",
    )

    # チェーンの作成
    chain = prompt | llm

    # ユーザーからの入力を受け取る
    topic = input("トピックを入力してください: ")

    # チェーンの実行
    result = chain.invoke({"topic": topic})

    # 結果の出力
    print("\n=== 生成結果 ===")
    print(result.content)


if __name__ == "__main__":
    main()
