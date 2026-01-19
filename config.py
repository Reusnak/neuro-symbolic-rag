import os

# --- 1. 路径配置 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 你的 Obsidian 仓库路径 (WSL 格式)
VAULT_PATH = "/mnt/c/Users/28232/Documents/Obsidian Vault"

# 数据持久化路径 (建议使用绝对路径以防万一)
PERSIST_DIR = os.path.join(BASE_DIR, "storage_data")
DB_PATH = os.path.join(PERSIST_DIR, "chroma_db")
DOC_STORE_PATH = os.path.join(PERSIST_DIR, "doc_store")
GRAPH_PATH = os.path.join(PERSIST_DIR, "knowledge_graph.pkl")
BM25_PATH = os.path.join(PERSIST_DIR, "bm25.pkl")

# 忽略的目录
IGNORE_DIRS = {".obsidian", ".trash", ".git", ".idea", "node_modules"}

# --- 2. 网络与模型配置 ---
try:
    # 获取 WSL 宿主机 IP
    WINDOWS_IP = os.popen("ip route | grep default | awk '{print $3}'").read().strip()
    if not WINDOWS_IP: WINDOWS_IP = "127.0.0.1"
except:
    WINDOWS_IP = "127.0.0.1"

OLLAMA_BASE_URL = f"http://{WINDOWS_IP}:11434"

# 模型名称
EMBED_MODEL_NAME = "nomic-embed-text"
LLM_MODEL_NAME = "qwen2.5:7b"

# --- 核心：补全 Reranker 物理路径 (报错就是因为少了这一行) ---
RERANKER_MODEL_PATH = "/home/reusnak/neuro-symbolic-rag/models/bge-reranker-v2-m3"

# --- 3. 检索参数 ---
RETRIEVAL_K = 10     # 向量检索初步召回数量
RERANK_TOP_K = 3     # 最终提供给 LLM 的上下文数量