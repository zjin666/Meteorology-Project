import numpy as np
import faiss
import re
import os
from openai import OpenAI
from config.load_key import load_key
from typing import List, Tuple

client = OpenAI(
    api_key=load_key("BAILIAN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

class TextVectorizer:
    def __init__(self):
        """初始化文本向量化器"""
        self.model = "text-embedding-v4"
        self.sentences = []
        self.embeddings = []
        self.index = None
        self.max_chunks = 1
        
    def split_text(self, file: str, rule: str) -> List[List[str]]:
        """将长文本分割成句子块"""
        with open(file, 'r', encoding='utf-8') as text:
            content = text.read()
        textList = []
        for chunk in re.split(rule, content):
            textList.append(chunk)
        return textList

    
    def encode_texts(self, texts: List[str]) -> List[float]:
        """将文本列表转换为向量"""
        embeddings = client.embeddings.create(
            model="text-embedding-v4",
            input=texts
        )
        info = embeddings.to_dict()
        return info["data"][0]["embedding"]
    
    def build_index(self, file: str, seperation: str):
        """构建FAISS索引并优化存储结构"""
        file_path = os.path.join(os.path.dirname(__file__), file)
        try :
            all_chunks = self.split_text(file_path, seperation)
            for chunk in all_chunks:
                if not chunk:
                    all_chunks.remove(chunk)
            self.sentences = all_chunks
            print(f"总共分割成 {len(all_chunks)} 个文本块")
            self.max_chunks = len(all_chunks)
            n_clusters = int(np.floor(np.sqrt(len(all_chunks))))
        
            for chunk in all_chunks:
                self.embeddings.append(np.array(self.encode_texts(chunk)))
            self.embeddings = np.array(self.embeddings)
            dimension = self.embeddings.shape[1]

            # 使用IVF索引实现聚类存储优化
            # 先进行K-means聚类确定中心点
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, n_clusters, faiss.METRIC_L2)
        
            self.index.train(self.embeddings)

            self.index.add(self.embeddings)
        
            print(f"FAISS索引构建完成，包含 {self.index.ntotal} 个向量")
        except Exception as e: print(f"错误: {str(e)}")
        
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """搜索相似文本"""
        query_vector = np.array(self.encode_texts([query])).reshape(1, -1)
        if k > self.max_chunks: k = self.max_chunks
        try:
            distances, indices = self.index.search(query_vector, k= k)
            results = []
            for i in range(k):
                idx = indices[0][i]
                distance = distances[0][i]
                if idx != -1:
                    results.append((self.sentences[idx], distance))
                
            return results
        except Exception as e: print(f"错误: {str(e)}") 




def main():
    # 示例使用
    vectorizer = TextVectorizer()
    
    # 示例文本数据
    texts = "test.txt"
    
    print("开始构建文本向量索引...")
    vectorizer.build_index(texts, '\n\n')
    
    # 测试搜索功能
    query = "未来世界在战后怎么样了？"
    print(f"\n查询: {query}")
    results = vectorizer.search(query, k=3)
    output = []
    
    print("\n相似文本结果:")
    for i, (text, distance) in enumerate(results, 1):
        output.append((f"{distance:.4f}", text))
        print(f"{i}. 距离: {distance:.4f}")
        print(f"   内容: {text}\n")
    print(output)

if __name__ == "__main__":
    main()
