from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
import json

# 環境変数の読み込み
load_dotenv()


# 出力の構造を定義
class Achievement(BaseModel):
    year: str = Field(description="年(ex. 1970)")
    overview: str = Field(description="内容")


class Profile(BaseModel):
    name: str = Field(description="人物の名前")
    lifespan: str = Field(description="人物の生存期間(ex. 1970~2000)")
    overview: str = Field(description="人物の概要")
    achievements: List[Achievement] = Field(description="人物の実績・功績")


def main():
    # LLMの初期化
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

    # Output Parser
    parser = PydanticOutputParser(pydantic_object=Profile)

    # Prompt Template
    prompt = PromptTemplate(
        input_variables=["person"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
        template="""
{person}について、中学校の教科書に記載するつもりで説明してください。

{format_instructions}
""",
    )

    chain = prompt | llm | parser

    person = input("人物名を入力してください: ")

    profile: Profile = chain.invoke({"person": person})

    print(json.dumps(profile.model_dump(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
