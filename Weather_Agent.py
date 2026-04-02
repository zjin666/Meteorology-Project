import sys
import os
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from config.load_key import load_key
from typing import List, Tuple
llm = ChatOpenAI(
    model = "qwen3.5-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key = load_key("BAILIAN_API_KEY"),
    temperature= 0.1,
)
from langchain.agents import create_agent
from QweatherTool import WeatherAPITool
tool_instance = WeatherAPITool(api_key=load_key("QWEATHER_API_KEY"))

@tool
def get_city_info(city_keyword: str) -> str:
    """
        根据城市关键词查询地理信息（ID、经纬度、行政区划）

        :param city_keyword: 城市名称，如 "杭州"
        :return: 包含城市信息的字典或错误信息
        """
    result = tool_instance.get_city_info(city_keyword)
    return str(result)

@tool
def get_weather_now(location_id: str) -> str:
    """
        获取指定城市的实时天气，包括了气温，湿度和风力，分别储存在"data"里

        :param location_id: 城市ID，由 get_city_info 返回
        :return: 实时天气数据
        """
    result = tool_instance.get_weather_now(location_id)
    return str(result)

@tool
def get_weather_daily(location_id: str, days: str = "3d") -> str:
    """
        获取指定城市的多日天气预报（支持 3d / 7d / 10d / 15d）

        :param location_id: 城市ID
        :param days: 预报天数，默认 "3d"
        :return: 多日天气数据
        """
    result = tool_instance.get_weather_daily(location_id, days)
    return str(result)

@tool
def get_weather_hourly(location_id: str, hours: str = "3d") -> str:
    """
        获取指定城市的逐小时天气预报（支持 24h / 72h / 168h）

        :param location_id: 城市ID
        :param hours: 预报时长，默认 "24h"
        :return: 逐小时天气数据
        """
    result = tool_instance.get_weather_hourly(location_id, hours)
    return str(result)

@tool
def get_rain(location_latitude: str, location_longitude: str) -> str:
    """
        获取指定经纬度的分钟级降水预报

        :param latitude: 纬度，字符串格式，如 "30.2741"
        :param longitude: 经度，字符串格式，如 "120.1541"
        :return: 降水预报数据
        """
    result = tool_instance.get_rain_forecast(location_latitude, location_longitude)
    return str(result)

from text_vectorizer import TextVectorizer
@tool
def RAG_Searcher(query: str, amount: int, seperation: str, original_Text: str = "Reference.txt") -> List[str]:
    """
        根据原本的资料分割并向量化然后储存，并根据给的问题获取一定数量的从最匹配依次到第n = amount个匹配的资料的分割片段
        （根据提供的资料判断什么分割符号最适合该文本，然后再将这些分割符号输入给seperation）

        :param original_Text: 资料的文本
        :param query: 用户的问题
        :param amount: 需要的匹配的资料分割的片段
        :param seperation: 分隔符号（以string的形式输入）
        :return 匹配资料的内容，以list的形式传递
    """
    vectorizer = TextVectorizer()
    vectorizer.build_index(original_Text, seperation)
    results = vectorizer.search(query, k=amount)
    output = []
    
    for i, (text, distance) in enumerate(results, 1):
        output.append(text)

    return "\n\n".join(output)



Weather_Tools = [get_city_info, get_weather_now, get_weather_daily, get_weather_hourly, get_rain, RAG_Searcher]
agent = create_agent(
    model=llm,
    tools= Weather_Tools,
    system_prompt=f"""你是一个智能天气助手，请借助{Weather_Tools}里的天气工具根据用户提出的问题回答提供对应的数据，
    你拥有资料库，它的文本名字是"Reference.txt"。请合理运用提供的工具来对用户的问题进行比对看看能不能再资料库里查找到适合的答案来丰富你的回答，但不要动用资料库以外的数据来回答这些内容
    生成的答案是以数据在前，然后再是你的结论和一些需要用户知道注意的内容，如果没有具体数据则直接说出结论和用户需要注意知道的内容

    （自动完成以下流程：用户提问 → 提取地点 → 调用 get_city_info → 获取编码，经纬度等必要信息 → 如需要调用其它必要的工具 → （可选）调用资料库查找匹配问题的内容 → 生成回答）""",
)

print("\n" + "=" * 50)
print("天气交互式查询 (输入 'quit' 退出):")
while True:
    user_input = input("\n请输入问题: ")
    if user_input.lower() == 'quit':
        break
    try:
        response = agent.invoke({"messages":[{"role":"user","content":user_input}]})
        print(f"回答: {response["messages"][-1].content}")
    except Exception as e:
        print(f"错误: {str(e)}")