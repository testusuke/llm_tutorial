from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()


def multiply_numbers(a: int, b: int) -> int:
    """2つの数字を掛け算する関数"""
    return a * b


def main():
    # LLMの初期化
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

    # 掛け算ツールの作成
    multiply_tool = Tool(
        name="multiply",
        func=multiply_numbers,
        description="2つの数字を掛け算します。入力は2つの整数で、出力はその積です。",
    )

    # プロンプトテンプレートの作成
    prompt = PromptTemplate(
        input_variables=["question"],
        template="""以下の質問に答えてください。必ず計算ツールを使用してください。

質問: {question}

計算ツールの使用方法：
1. 必ずツールを使用して計算を行ってください
2. 計算結果を「計算: [計算内容]」の形式で示してください
3. 最終的な答えを「答え: [計算結果]」の形式で示してください
""",
    )

    # チェーンの作成
    chain = prompt | llm.bind_tools([multiply_tool])

    # ユーザーからの入力を受け取る
    question = input("計算したい内容を入力してください: ")

    # チェーンの実行
    result = chain.invoke({"question": question})

    # 結果の出力
    print("\n=== 計算結果 ===")
    print(result.content)

    # ツールの使用状況を確認
    if hasattr(result, "tool_calls"):
        print("\n=== ツールの使用状況 ===")
        for tool_call in result.tool_calls:
            print(f"使用されたツール: {tool_call['name']}")
            print(f"ツールの引数: {tool_call['args']}")


if __name__ == "__main__":
    main()
