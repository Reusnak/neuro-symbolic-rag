import os
import shutil
import pickle
import uuid
import networkx as nx
from typing import Iterator, List, Optional, Sequence, Tuple

# 核心依赖 (适配 LangChain 0.3+)
from langchain_core.stores import ByteStore
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever

import config
from splitter import TextSplitterFactory

# --- 1. 轻量化本地存储存储父文档 ---
class LocalFileStore(ByteStore):
    def __init__(self, root_path: str):
        self.root_path = root_path
        os.makedirs(root_path, exist_ok=True)

    def mget(self, keys: Sequence[str]) -> List[Optional[bytes]]:
        results = []
        for k in keys:
            path = os.path.join(self.root_path, k)
            results.append(open(path, "rb").read() if os.path.exists(path) else None)
        return results

    def mset(self, key_value_pairs: Sequence[Tuple[str, bytes]]) -> None:
        for k, v in key_value_pairs:
            with open(os.path.join(self.root_path, k), "wb") as f:
                f.write(v)

    def mdelete(self, keys: Sequence[str]) -> None:
        for k in keys:
            path = os.path.join(self.root_path, k)
            if os.path.exists(path): os.remove(path)

    def yield_keys(self, prefix: Optional[str] = None) -> Iterator[str]:
        for k in os.listdir(self.root_path):
            if prefix is None or k.startswith(prefix): yield k

# --- 2. 手写父子文档管理逻辑 ---
class SimpleParentRetriever:
    def __init__(self, vectorstore, docstore, child_splitter, parent_splitter):
        self.vectorstore = vectorstore
        self.docstore = docstore
        self.child_splitter = child_splitter
        self.parent_splitter = parent_splitter
        self.id_key = "doc_id"

    def add_documents(self, documents: List[Document]):
        for doc in documents:
            # 生成语义完整的父块 (用于最终阅读)
            parent_docs = self.parent_splitter.split_documents([doc])
            for p_doc in parent_docs:
                _id = str(uuid.uuid4())
                # 存储原始父块
                self.docstore.mset([(_id, pickle.dumps(p_doc))])
                # 生成细颗粒度子块 (用于精准匹配)
                child_docs = self.child_splitter.split_documents([p_doc])
                for c_doc in child_docs:
                    c_doc.metadata[self.id_key] = _id
                self.vectorstore.add_documents(child_docs)

# --- 3. 存储管理器 (核心) ---
class StorageManager:
    def __init__(self):
        # 初始化 Embedding
        self.embedding = OllamaEmbeddings(
            model=config.EMBED_MODEL_NAME,
            base_url=config.OLLAMA_BASE_URL
        )
        self.splitter_factory = TextSplitterFactory()
        # 确保数据目录存在
        self.docstore = LocalFileStore(config.DOC_STORE_PATH)
        os.makedirs(config.PERSIST_DIR, exist_ok=True)

    def clear_data(self):
        """清空所有本地索引数据"""
        if os.path.exists(config.PERSIST_DIR):
            shutil.rmtree(config.PERSIST_DIR)
        os.makedirs(config.PERSIST_DIR, exist_ok=True)

    # --- 修复：补全缺失的 load_graph 方法 ---
    def load_graph(self) -> nx.Graph:
        """从本地持久化文件加载知识图谱"""
        if os.path.exists(config.GRAPH_PATH):
            try:
                with open(config.GRAPH_PATH, "rb") as f:
                    print(f"🕸️ 正在加载知识图谱: {config.GRAPH_PATH}")
                    return pickle.load(f)
            except Exception as e:
                print(f"⚠️ 图谱加载失败: {e}, 返回空图")
                return nx.Graph()
        else:
            print("⚠️ 未发现图谱文件，初始化新图谱")
            return nx.Graph()

    def save_graph(self, graph: nx.Graph):
        """将知识图谱保存到本地"""
        with open(config.GRAPH_PATH, "wb") as f:
            pickle.dump(graph, f)
            print(f"✅ 图谱已保存至: {config.GRAPH_PATH}")

    def build_vector_bm25_index(self, docs: List[Document]):
        """构建双路索引：Chroma(向量) + BM25(关键词)"""
        # 1. 初始化 Chroma
        vectorstore = Chroma(
            collection_name="rag_collection",
            embedding_function=self.embedding,
            persist_directory=config.DB_PATH
        )
        
        # 2. 初始化父文档存储 (ByteStore)
        doc_store = LocalFileStore(config.DOC_STORE_PATH)
        
        # 3. 运行父子文档切分逻辑
        retriever = SimpleParentRetriever(
            vectorstore=vectorstore, 
            docstore=doc_store,
            child_splitter=self.splitter_factory.get_child_splitter(),
            parent_splitter=self.splitter_factory.get_parent_splitter()
        )

        print(f"💾 正在向量化并索引 {len(docs)} 个原始文档...")
        retriever.add_documents(docs)

        # 4. 构建并保存 BM25 索引
        print("🧮 正在构建 BM25 关键词索引...")
        all_data = vectorstore.get()
        if all_data['documents']:
            bm25_docs = [
                Document(page_content=d, metadata=m) 
                for d, m in zip(all_data['documents'], all_data['metadatas'])
            ]
            bm25_retriever = BM25Retriever.from_documents(bm25_docs)
            with open(config.BM25_PATH, "wb") as f:
                pickle.dump(bm25_retriever, f)
        print("✅ 存储层构建成功！")

    def get_retriever_components(self):
        """为前端提供检索所需的全部物理组件"""
        # 加载向量库
        vectorstore = Chroma(
            collection_name="rag_collection", 
            persist_directory=config.DB_PATH, 
            embedding_function=self.embedding
        )
        # 加载父文档库
        doc_store = LocalFileStore(config.DOC_STORE_PATH)
        
        # 加载 BM25 (如果存在)
        bm25 = None
        if os.path.exists(config.BM25_PATH):
            try:
                with open(config.BM25_PATH, "rb") as f:
                    bm25 = pickle.load(f)
            except Exception as e:
                print(f"⚠️ BM25 加载失败: {e}")
                
        return vectorstore, doc_store, bm25