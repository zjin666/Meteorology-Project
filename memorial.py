import os
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.tools import tool
from config.load_key import load_key
from typing import List
import sqlite3

llm = ChatOpenAI(
    model = "qwen3.5-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key = load_key("BAILIAN_API_KEY"),
    temperature= 0,
)

class MemorySystem:
    def __init__(self, storage_path="Memories.db"):
        self.memories = []
        self.storage_path = storage_path
        self.load_memories()
    
    def check_data_exists(self, db_path: str, table_name: str, column_name: str, value: str):
        """
        使用EXISTS子查询检查数据是否存在
        Args:
            db_path: 数据库文件路径
            table_name: 表名
            column_name: 列名
            value: 要查询的值
        Returns:
        bool: 数据存在返回True，否则返回False
        """
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
        
            query = f"""
            SELECT EXISTS (
                SELECT 1 
                FROM {table_name} 
                WHERE {column_name} = ?
            ) AS data_exists
            """
            cursor.execute(query, (value,))
            result = cursor.fetchone()
        
            return bool(result[0]) if result else False
        except sqlite3.Error as e:
            print(e)
            return False
        finally:
            if conn:
                conn.close()

    def load_memories(self):
        """从对应的表里加载已有的记忆"""
        if os.path.exists(self.storage_path):
            self.memories.clear()

            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()
            cursor.execute("""SELECT name FROM sqlite_master WHERE type='table'""")
            tables = cursor.fetchall()

            for table in tables:
                table_name = table[0]
                cursor.execute(f"""SELECT * FROM {table_name}""")
                contents = cursor.fetchall()
                for content in contents:
                    self.memories.append({
                    "table": table_name,
                    "question": content[1],
                    "summary": content[2],
            })
            conn.close()

    def save_memories(self, change: bool, table_name: str):
        """将记忆保存到相应表里，如果change为True，查看是否有同样的类似的问题，然后修改它的summary"""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()
        cursor.execute(f'''CREATE TABLE IF NOT EXISTS {table_name} (
                table_name TEXT,
                user_message TEXT,
                summary TEXT)''')
        conn.commit()
        try:
            for item in self.memories:
                if not self.check_data_exists(self.storage_path, table_name, 'user_message', item["question"]) and item["table"] == table_name:
                    cursor.execute(f'''
                        INSERT INTO {table_name} VALUES (:table_name, :user_message, :summary)
                    ''', dict(table_name=item["table"], user_message=item["question"], summary=item["summary"]))
                    print(f"[MEMORY STORED IN TABLE: {table_name}]")
            if change:
                question = self.memories[-1]["question"]
                summary = self.memories[-1]["summary"]
                if self.check_data_exists(self.storage_path, table_name, 'user_message', question):
                    cursor.execute(f"""UPDATE {table_name} SET summary=:summary WHERE user_message=:question """, dict(summary=summary, question=question))
                    print(f"[MEMORY CHANGED IN TABLE: {table_name}]")
            conn.commit()
        except sqlite3.Error as e:
            print(f"Error exist: {e}")
        finally:
            if conn:
                conn.close()

    def summarize_text(self, text: str) -> str:
        """使用LangChain的摘要链对文本进行摘要"""
        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            docs = [Document(page_content=t) for t in text_splitter.split_text(text)]
        
            template = """
            请为以下文本生成摘要，摘要长度不超过{max_length}个字：
    
            {document_content}
    
            摘要：
            """
    
            prompt = PromptTemplate(
                input_variables=["document_content", "max_length"],
                template=template
            )
            chain = prompt | llm
            reply = chain.invoke({
                "document_content": docs,
                "max_length": 100
            })
            return reply.content
        except Exception as e: print(f"Error: {e}")

    def store_memory(self, question: str, answer: str, table_name: str, should_store: bool = True, should_change: bool = False):
        """存储记忆，根据should_store参数决定是否存储，根据should_change决定是否改变记忆里出现的类似对话的summary"""
        if not should_store:
            print("用户明确指示不储存此记忆")
            return
            
        try:
            memory_entry = f"Question: {question}\nAnswer: {answer}"
            summary = self.summarize_text(memory_entry)
            self.memories.append({
                "table": table_name,
                "question": question,
                "summary": summary
            })
            self.save_memories(should_change, table_name)
        except Exception as e: print(f"E!错误: {e}")

    def get_memories(self):
        """读取已经拥有的记忆"""
        return self.memories

    def clear_memories(self, indices: List[int], table_name: str):
        """删除已有的记忆，如果{clear all}是true则删除所有的记忆，反之根据index删除指定的记忆"""
        print("开始删除")
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()
        query = f"DROP TABLE IF EXISTS {table_name}"
        for index in set(indices):
            if (index != -1): 
                question = self.memories[index]["question"]
                cursor.execute(f"""DELETE FROM {table_name} WHERE user_message=?""", (question,))
            else: 
                cursor.execute(query)
                break
        conn.commit()
        conn.close()

                

@tool
def memory_System(user_message: str, system_reply: str, name: str, allow_store: bool, delete_indecies: List[int] = []) -> str:
    """记忆存储删除的工具，根据{allow_store}判断是否需要存储记忆，
    判断是否需要删除所有记忆，如果要求删除所有记忆，{delete_index}输入-1。如果不要求删除所有记忆，则根据用户问题判断是记忆里的那句话，把那句话的index输入给{delete_index}
    如果要删除记忆，先用get_tables获取表的名称，再根据名称删除，不要自己创造不存在的表的名称
        user_message:user最近一条的问题，需要是str
        system_reply:最近一条问题的回答，需要时str
        delete_indecies：根据用户需要删除的内容决定放入记忆里这些内容对应的index(如果要删除整个表赋值-1)，然后输入，这是一个Set，里面的内容得是int
        name: 若get_tables的output里已有表，则根据问题判断是否要创建一个新名称，要是意思相近就不要新建，改用已有的表的名称，若没有则根据问题创建一个名称，要求是str
    """
    memory_system = MemorySystem()
    if (delete_indecies): 
        memory_system.clear_memories(indices=delete_indecies,table_name=name)
        return "Chat Memories Cleared"
    memory_system.store_memory(question=user_message,answer=system_reply,should_store=allow_store,table_name=name)
@tool 
def get_tables() -> List[str]:
    """读取存在的表，输出目前存在的表"""
    memory_system = MemorySystem()
    all_tables = []
    for item in memory_system.get_memories():
        all_tables.append(item["table"])
    return all_tables
@tool 
def get_history():
    """读取存在的记忆，若返回值为空则直接说没有聊天记录"""
    memory_system = MemorySystem()
    if not memory_system.get_memories():
        return "no history"
    return memory_system.get_memories()


agent = create_agent(
    model=llm,
    tools= [get_tables,memory_System,get_history],
    system_prompt="""你是一个聊天助手，运用工具来存储记忆，读取记忆和删除记忆
    如果用户没有明确要求不记录聊天，默认自动存储每一句问答，用户要求读取和删除之前的记忆时也要使用工具来完成用户的要求
        使用memory_System之前一定要先收集一下get_tables里的表的名称""",
)

def main():
    while True:
        user_input = input("\n请输入问题: ")
        if user_input.lower() == 'quit':
            break
        try:
            response = agent.invoke({"messages":[{"role":"user","content":user_input}]})
            print(f"回答: {response["messages"][-1].content}")
        except Exception as e:
            print(f"错误: {str(e)}")

if __name__ == "__main__":
    main()