# 职责：提供 Markdown 语义切分与父子块切分逻辑。
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

class TextSplitterFactory:
    def __init__(self):
        # 定义 Markdown 标题层级
        self.headers_to_split_on = [("#", "H1"), ("##", "H2"), ("###", "H3")]

    def get_parent_splitter(self):
        """父块：提供给 LLM 的完整语义上下文 (1000字符)"""
        return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    def get_child_splitter(self):
        """子块：用于精确向量匹配 (200字符)"""
        return RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    
    def pre_split_markdown(self, docs):
        """
        预处理：优先按 Markdown 标题切分，
        让元数据中带上 'H1', 'H2' 等上下文信息。
        """
        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=self.headers_to_split_on)
        split_docs = []
        
        for doc in docs:
            if doc.metadata.get("type") == ".md":
                # 按标题切分，增强搜索精度
                chunks = md_splitter.split_text(doc.page_content)
                for c in chunks:
                    c.metadata.update(doc.metadata) # 继承文件名等信息
                    split_docs.append(c)
            else:
                # PDF 等非 MD 文件直接进入下一阶段
                split_docs.append(doc)
                
        return split_docs