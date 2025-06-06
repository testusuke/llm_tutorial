from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()


def main():
    # LLMの初期化
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

    question = input("質問を入力してください: ")

    # LLMを呼び出す
    result = llm.invoke(question)

    # 結果の出力
    print("\n=== 生成結果 ===")
    print(result.content)


if __name__ == "__main__":
    main()
