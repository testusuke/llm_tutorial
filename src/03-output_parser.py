from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()


# 出力の構造を定義
class TopicAnalysis(BaseModel):
    main_points: List[str] = Field(description="トピックの主要なポイントのリスト")
    summary: str = Field(description="トピックの簡潔な要約")


def main():
    # LLMの初期化
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

    # Parserの初期化
    parser = PydanticOutputParser(pydantic_object=TopicAnalysis)

    # プロンプトテンプレートの作成
    prompt = PromptTemplate(
        input_variables=["topic"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
        template="""以下のトピックについて分析してください：{topic}

{format_instructions}

回答は必ず指定された形式で出力してください。""",
    )

    # チェーンの作成
    chain = prompt | llm | parser

    # ユーザーからの入力を受け取る
    topic = input("トピックを入力してください: ")

    # チェーンの実行
    result = chain.invoke({"topic": topic})

    # 結果の出力
    print("\n=== 生成結果 ===")
    print(f"主要なポイント:")
    for i, point in enumerate(result.main_points, 1):
        print(f"{i}. {point}")
    print(f"\n要約: {result.summary}")


if __name__ == "__main__":
    main()
