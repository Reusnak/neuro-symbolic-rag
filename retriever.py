import torch
import pickle
import os
import networkx as nx
import sys
import importlib.util

# --- 1. 绝对物理路径注入 (处理 langchain_classic 兼容性) ---
ENSEMBLE_PATH = "/home/reusnak/neuro-symbolic-rag/.venv/lib/python3.12/site-packages/langchain_classic/retrievers/ensemble.py"
SITE_PACKAGES = "/home/reusnak/neuro-symbolic-rag/.venv/lib/python3.12/site-packages"

if SITE_PACKAGES not in sys.path:
    sys.path.insert(0, SITE_PACKAGES)

try:
    spec = importlib.util.spec_from_file_location("ensemble_fixed", ENSEMBLE_PATH)
    ensemble_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ensemble_module)
    EnsembleRetriever = ensemble_module.EnsembleRetriever
except:
    from langchain.retrievers import EnsembleRetriever

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import config
from storage import StorageManager

class RAGRetriever:
    def __init__(self):
        print("⚙️ 正在初始化多模态检索引擎...")
        self.storage = StorageManager()
        
        # 获取检索组件 (vectorstore, docstore, bm25)
        components = self.storage.get_retriever_components()
        # --- 变量名对齐：确保使用的是 self.docstore ---
        self.vectorstore, self.docstore, self.bm25 = components
            
        self.graph = self.storage.load_graph()
        
        # 1. 初始化 Reranker (精排模型)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = os.path.basename(config.RERANKER_MODEL_PATH)
        print(f"📥 加载重排序模型: {model_name} (设备: {device})")
        
        try:
            self.reranker = HuggingFaceCrossEncoder(
                model_name=config.RERANKER_MODEL_PATH,
                model_kwargs={'device': device}
            )
        except Exception as e:
            print(f"⚠️ GPU 加载重排序模型失败，回退到 CPU: {e}")
            self.reranker = HuggingFaceCrossEncoder(
                model_name=config.RERANKER_MODEL_PATH,
                model_kwargs={'device': 'cpu'}
            )

        # 2. 组合检索器 (混合召回：向量 + 关键词)
        if self.vectorstore:
            self.child_retriever = self.vectorstore.as_retriever(search_kwargs={"k": config.RETRIEVAL_K})
            
            if self.bm25:
                self.bm25.k = config.RETRIEVAL_K
                self.ensemble = EnsembleRetriever(
                    retrievers=[self.bm25, self.child_retriever], 
                    weights=[0.3, 0.7] 
                )
            else:
                print("⚠️ 未发现 BM25 索引，仅使用向量检索。")
                self.ensemble = self.child_retriever
        else:
            self.ensemble = None

    def _get_parent_content(self, child_docs):
        """还原父文档：使用 self.docstore (对齐 StorageManager)"""
        parent_ids = list({d.metadata.get("doc_id") for d in child_docs if "doc_id" in d.metadata})
        if not parent_ids: 
            return []
        
        # --- 关键修改：确保变量名是 docstore ---
        bytes_data = self.docstore.mget(parent_ids)
        return [pickle.loads(b) for b in bytes_data if b]

    def _graph_enhance(self, source_name, seen_sources):
        """图谱增强：寻找 Obsidian 中的双链关联"""
        if self.graph is None or not self.graph.has_node(source_name): 
            return ""
        neighbors = [n for n in self.graph.neighbors(source_name) if n not in seen_sources]
        if not neighbors: 
            return ""
        return f"\n   [💡 关联笔记建议]: {', '.join(neighbors[:3])}"

    def search(self, query):
        """核心检索流程：混合召回 -> 父块映射 -> 重排序 -> 图谱增强"""
        if self.ensemble is None:
            return "❌ 系统尚未初始化，请先运行数据注入脚本。"

        # (1) 混合召回子文档块
        child_docs = self.ensemble.invoke(query)
        
        # (2) 映射回具有完整语义的父文档
        parents = self._get_parent_content(child_docs)
        if not parents: 
            return "未找到相关背景知识。"

        # (3) 重排序 (Rerank)：解决 predict 属性丢失问题
        pairs = [[query, doc.page_content] for doc in parents]
        
        try:
            # 兼容性调用：LangChain 0.3+ 可能会封装底层模型
            if hasattr(self.reranker, 'model') and hasattr(self.reranker.model, 'predict'):
                scores = self.reranker.model.predict(pairs)
            elif hasattr(self.reranker, 'predict'):
                scores = self.reranker.predict(pairs)
            else:
                # 最后的保底方案
                print("⚠️ 无法在 Reranker 上找到 predict 方法，尝试直接调用底层的 client")
                scores = self.reranker.client.predict(pairs)
        except Exception as e:
            print(f"⚠️ 重排序失败: {e}，将按原始顺序排列")
            scores = [1.0] * len(parents)

        ranked = sorted(zip(parents, scores), key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, score in ranked[:config.RERANK_TOP_K]]

        # (4) 组装最终上下文
        context_parts = []
        seen_sources = set()
        for doc in top_docs:
            src = doc.metadata.get("source", "未知")
            seen_sources.add(src)
            
            h1 = doc.metadata.get("H1", "")
            h2 = doc.metadata.get("H2", "")
            path_info = f" -> {h1}" if h1 else ""
            if h2: path_info += f" -> {h2}"

            header = f"【来源: {src}{path_info}】"
            graph_info = self._graph_enhance(src, seen_sources)
            
            context_parts.append(f"{header}\n{doc.page_content}{graph_info}")

        return "\n\n".join(context_parts)