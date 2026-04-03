import re
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from config.load_key import load_key
from langchain.agents import create_agent


execute = ChatOpenAI(
    model = "deepseek-v3.2",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key = load_key("BAILIAN_API_KEY"),
    temperature= 0,
)

llm = ChatOpenAI(
    model = "qwen-flash",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key = load_key("BAILIAN_API_KEY"),
    temperature= 0,
)

reflect = ChatOpenAI(
    model = "qwq-32b-preview",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key = load_key("BAILIAN_API_KEY"),
    temperature= 0,
)


class AgentState(BaseModel):
    question: str
    answer: str = ""
    initial_ans: str = ""
    reflection_count: int = 0
    reason: str = ""
    next_step: str = ""
    next_node: str = ""

def executor_node(state: AgentState):
    print(f"[执行器] 接收问题: {state.question}")
    agent = create_agent(
    model=execute,
    tools= [],
    system_prompt=f"""你是一个对话助手，请根据用户问题回答，如果{state.reason}不为空，则根据其中“建议”后的内容来辅助你的回答，这些信息会帮助你如何更好提升对用户的问题回答的质量
    """,
    )
    response = agent.invoke({"messages":[{"role":"user","content":state.question}]})
    result = response["messages"][-1].content
    print(f"[执行器] 准备传输")
    if (state.initial_ans == ""):
        return {"answer": result, "initial_ans": result, "next_step": "前往reflector进行校验"}
    return {"answer": result, "next_step": "前往reflector进行校验"}

def intermediate_node(state: AgentState):
    print(f"[中转器] 接收到信息")
    print(state.next_step)
    template = """
            请严格根据next_step的内容，也就是{next}来输出以下内容：
                如果内容的意愿里是前往reflector，则输出: "reflector"
                如果内容的意愿里是前往END，则输出: "END"
                如果内容的意愿里是前往executor，则输出: "executor"
                不用做其他思考
            """
    prompt = PromptTemplate(
                input_variables=["next", "count"],
                template=template
            ) 
    chain = prompt | llm
    response = chain.invoke({
                "next": state.next_step,
                "count": state.reflection_count,
            }).content
    Letters = ''.join(re.findall(r'[a-zA-Z]', response))
    return {"next_node":Letters}

def route_from_intermediate(state: AgentState) -> str:
    if (state.reflection_count >= 5 and state.next_node != "END"):
        print("[中转器] 超出最大循环数，强制承接到END")
        return "END"
    else: print(f"[中转器] 承接到{state.next_node}中...") 
    return state.next_node

def reflector_node(state: AgentState):
    count = state.reflection_count + 1
    print(f"[反思器] 承接成功，第{count}次校对中...")

    question = state.question
    answer = state.answer

    template1 = """
            请以string格式返回输出
            
            格式要求（无需其他内容）：
            包含 分数 和 建议 两个板块，
            分数：一个1到10的数字，越大代表{answer}作为对{question}的解答越合理
            建议：根据 分数 给出改进的建议，不超过50字
            """
    prompt = PromptTemplate(
                input_variables=["question", "answer"],
                template=template1
            ) 
    chain = prompt | reflect
    reply = chain.invoke({
                "question": question,
                "answer": answer,
            }).content

    template2 = """
            请以string格式返回输出:

            格式要求（无需其他内容）：
                回答只需是“前往executor”或者“前往END”，如果{reply}里的分数的数字小于8则输出: 前往executor 否则输出: 前往END
            
            """
    prompt2 = PromptTemplate(
                input_variables=["reply"],
                template=template2
            ) 
    chain2 = prompt2 | llm
    reply2 = chain2.invoke({"reply": reply}).content 
    print("[反思器] 完成校对")
    return {"reflection_count": count,"reason": reply, "next_step": reply2}

def create_workflow():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("executor", executor_node)
    workflow.add_node("reflector", reflector_node)
    workflow.add_node("transporter", intermediate_node)
    
    workflow.add_edge("executor", "transporter")
    workflow.add_edge("reflector", "transporter")
    workflow.add_conditional_edges(
        "transporter",
        route_from_intermediate,
        {
            "executor": "executor",
            "reflector": "reflector",
            "END": END,
        }
    )
    workflow.set_entry_point("executor")
    
    return workflow.compile()

# 主函数
def main():
    app = create_workflow()
    
    # 初始化状态
    initial_state = AgentState(question="玉鱼能游多快？")
    
    print("=== 启动Agent循环交互流程 ===")
    final_state = app.invoke(initial_state)
    
    print("\n=== 最终结果 ===")
    print(f"用户问题: {final_state["question"]}")
    print(f"初始答案: {final_state["initial_ans"]}")
    print(f"最终答案: {final_state["answer"]}")
    print(f"循环次数: {final_state["reflection_count"]}")

if __name__ == "__main__":
    main()
