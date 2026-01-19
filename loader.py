import os
import re
import networkx as nx
import fitz  
from tqdm import tqdm
from langchain_core.documents import Document
import config

class ContentLoader:
    def __init__(self):
        """只保留轻量级初始化"""
        pass

    def _extract_links(self, text):
        """解析 Obsidian 双链 [[Target]] 或 [[Target|Alias]]"""
        return re.findall(r'\[\[(.*?)\]\]', text)

    def _load_pdf(self, filepath):
        """使用 PyMuPDF 提取 PDF 文本层内容"""
        try:
            doc = fitz.open(filepath)
            text = "\n".join([page.get_text() for page in doc])
            doc.close()
            if not text.strip():
                print(f"⚠️ 跳过扫描版或无文本PDF: {os.path.basename(filepath)}")
            return text
        except Exception as e:
            print(f"❌ PDF 解析错误 {filepath}: {e}")
            return ""

    def load_vault(self):
        """遍历 Obsidian 库，构建文档列表与关系图谱"""
        docs = []
        graph = nx.Graph()
        
        if not os.path.exists(config.VAULT_PATH):
            raise FileNotFoundError(f"未找到路径: {config.VAULT_PATH}")

        # 1. 递归扫描文件
        all_files = []
        for root, dirs, files in os.walk(config.VAULT_PATH):
            dirs[:] = [d for d in dirs if d not in config.IGNORE_DIRS]
            for f in files:
                all_files.append(os.path.join(root, f))

        print(f"📂 扫描到 {len(all_files)} 个文件，正在解析文本...")

        # 2. 遍历解析
        for path in tqdm(all_files, desc="Parsing"):
            ext = os.path.splitext(path)[1].lower()
            name = os.path.basename(path).replace(ext, "")
            content = ""

            if ext == ".md":
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            elif ext == ".pdf":
                content = self._load_pdf(path)
            else:
                continue

            if not content.strip():
                continue

            # 3. 建立知识图谱节点与双链边
            graph.add_node(name, path=path, type=ext)
            for link in self._extract_links(content):
                target = link.split('|')[0] # 过滤别名
                graph.add_edge(name, target)

            # 4. 生成 LangChain 标准文档对象
            docs.append(Document(
                page_content=content, 
                metadata={"source": name, "type": ext}
            ))

        return docs, graph