# 职责：负责逻辑路由（Router）、查询重写（Rewriter）和最终答案生成（LLM）。
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import config

class RAGGenerator:
    def __init__(self):
        # 统一使用配置中的模型地址和名称
        self.llm = ChatOllama(
            model=config.LLM_MODEL_NAME, 
            base_url=config.OLLAMA_BASE_URL,
            temperature=0  # 路由和重写需要高确定性
        )

    def router(self, query: str) -> bool:
        """
        Agent 路由：判断是否需要检索本地笔记。
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个意图分类专家。判断用户输入是否需要查询其个人笔记或专业文档。"),
            ("human", "闲聊/简单问候 -> NO; 专业问题/具体事实/文档查询 -> YES。\n输入内容: {q}\n只输出 YES 或 NO:")
        ])
        chain = prompt | self.llm | StrOutputParser()
        try:
            result = chain.invoke({"q": query}).strip().upper()
            return "YES" in result
        except:
            return True # 默认检索以保证安全

    def rewriter(self, query: str) -> str:
        """
        查询重写：将口语化问题转化为更适合向量检索的专业关键词。
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个检索优化专家。请将用户的问题重写为 3-5 个适合检索的关键词组合。"),
            ("human", "去除无意义助词，补充隐含的领域上下文。\n原始问题: {q}\n优化后关键词:")
        ])
        chain = prompt | self.llm | StrOutputParser()
        # 清洗可能出现的引号或编号
        rewritten = chain.invoke({"q": query}).strip().replace('"', '').replace('1.', '')
        return rewritten

    def generate_stream(self, query: str, context: str):
        """
        最终生成：基于检索到的上下文进行流式回答。
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个知识库助手。请根据提供的【背景知识】诚实地回答问题。如果知识库中没有相关信息，请直接说明，严禁编造。"),
            ("human", "【背景知识】:\n{ctx}\n\n---\n【用户提问】: {q}")
        ])
        # 使用较低温度保证回答的稳定性
        chain = prompt | self.llm | StrOutputParser()
        return chain.stream({"ctx": context, "q": query})