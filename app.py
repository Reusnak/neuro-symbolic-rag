import sys
import os
import importlib.util
import streamlit as st

# --- 1. 绝对物理路径注入 ---
# 这是你刚才 find 出来的真实坐标
ENSEMBLE_PATH = "/home/reusnak/neuro-symbolic-rag/.venv/lib/python3.12/site-packages/langchain_classic/retrievers/ensemble.py"
SITE_PACKAGES = "/home/reusnak/neuro-symbolic-rag/.venv/lib/python3.12/site-packages"

# 强制将 site-packages 加入搜索路径
if SITE_PACKAGES not in sys.path:
    sys.path.insert(0, SITE_PACKAGES)

try:
    # 暴力加载：直接从物理文件读取
    spec = importlib.util.spec_from_file_location("ensemble_fixed", ENSEMBLE_PATH)
    ensemble_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ensemble_module)
    EnsembleRetriever = ensemble_module.EnsembleRetriever
    print("✅ 物理加载 EnsembleRetriever 成功")
except Exception as e:
    # 如果物理加载失败，尝试最后一种标准导入
    from langchain.retrievers import EnsembleRetriever

from langchain_ollama import ChatOllama
try:
    # 路径 A: 新版本的标准位置
    from langchain_core.messages import SystemMessage, HumanMessage
    print("✅ 通过 langchain_core 加载消息组件")
except ImportError:
    try:
        # 路径 B: 某些特定 0.3.x 版本的兼容位置
        from langchain.schema import SystemMessage, HumanMessage
    except ImportError:
        # 路径 C: 物理文件暴力加载 (最后的保底)
        import importlib.util
        # 这里的路径是 3.12 环境下的标准核心包位置
        core_path = "/home/reusnak/neuro-symbolic-rag/.venv/lib/python3.12/site-packages/langchain_core/messages/__init__.py"
        if os.path.exists(core_path):
            spec = importlib.util.spec_from_file_location("messages_fixed", core_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            SystemMessage = mod.SystemMessage
            HumanMessage = mod.HumanMessage
        else:
            st.error("❌ 无法定位 langchain_core。请运行: pip install langchain-core")
            st.stop()
from retriever import RAGRetriever
import config 

# --- 3. Streamlit 页面配置 ---
st.set_page_config(
    page_title="Neuro-Symbolic RAG", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧠 神经符号知识库")
st.caption("基于 Obsidian 笔记、图谱增强与混合检索的本地 AI 助手")

# --- 4. 资源初始化 (带缓存) ---
@st.cache_resource(show_spinner="正在加载 AI 模型与索引...")
def init_all():
    try:
        # 初始化检索引擎
        engine = RAGRetriever()
        # 初始化 LLM (Ollama)
        llm = ChatOllama(
            model=config.LLM_MODEL_NAME,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0.3
        )
        return engine, llm
    except Exception as e:
        st.error(f"❌ 初始化失败: {str(e)}")
        st.info("请确保已通过 scripts/ingest.py 构建了索引，并启动了 Ollama 服务。")
        st.stop()

engine, llm = init_all()

# --- 5. 聊天记录与侧边栏 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("⚙️ 系统状态")
    st.success("✅ 检索引擎: 就绪")
    st.info(f"🤖 当前模型: {config.LLM_MODEL_NAME}")
    if st.button("🗑️ 清空对话历史"):
        st.session_state.messages = []
        st.rerun()

# 渲染对话历史
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 6. 核心问答逻辑 ---
if query := st.chat_input("输入你的问题..."):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # 生成助手回答
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            # 第一步：执行检索 (Retrieval)
            with st.status("🔍 正在检索知识库...", expanded=False) as status:
                context = engine.search(query)
                status.update(label="✅ 检索完成", state="complete")

            # 第二步：构建消息序列
            messages = [
                SystemMessage(content=f"你是一个专业的 Obsidian 知识助手。请结合以下背景知识回答问题。\n\n背景知识：\n{context}"),
                HumanMessage(content=query)
            ]

            # 第三步：流式生成 (Streaming)
            for chunk in llm.stream(messages):
                full_response += chunk.content
                response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            
            # 保存到历史记录
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"⚠️ 生成回答时出错: {str(e)}")