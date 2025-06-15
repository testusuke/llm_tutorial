from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain_community.tools import TavilySearchResults
from langchain.agents import create_tool_calling_agent, AgentExecutor
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


def extract_person_info(llm: ChatOpenAI, input: str) -> Profile:
    # Output Parser
    parser = PydanticOutputParser(pydantic_object=Profile)

    # Prompt Template
    prompt = PromptTemplate(
        input_variables=["input"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
        template="""
次の文章から人物の情報を抽出してください。
# 人物の情報
{input}

# 出力形式
{format_instructions}
""",
    )

    chain = prompt | llm | parser

    profile: Profile = chain.invoke({"input": input})

    return profile


def research_person_info(llm: ChatOpenAI, person: str) -> str:
    tools = [TavilySearchResults(max_results=5)]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "次の人物について、生存期間、概要、実績や功績を詳しく調査してください。必要に応じて tavily_search_results_json を呼びます。",
            ),
            ("human", "人物: {input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return executor.invoke({"input": person})["output"]


def main():
    # LLMの初期化
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

    person = input("人物名を入力してください: ")

    report = research_person_info(llm, person)
    profile = extract_person_info(llm, report)

    print(json.dumps(profile.model_dump(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
