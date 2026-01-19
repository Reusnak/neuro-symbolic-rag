# 数据处理：串联 Loader, Splitter, Storage。
from loader import ContentLoader
from splitter import TextSplitterFactory
from storage import StorageManager

def main():
    print("🚀 启动数据入库流水线...")
    
    # 1. 初始化存储
    storage = StorageManager()
    storage.clear_data() # 如果需要增量更新，可以注释掉这行

    # 2. 加载文档 & 图谱
    loader = ContentLoader()
    raw_docs, graph = loader.load_vault()
    
    if not raw_docs:
        print("⚠️ 未找到文档，请检查 config.py 路径")
        return

    # 3. 保存图谱
    storage.save_graph(graph)

    # 4. 预处理 Markdown (按 Header 切分)
    splitter = TextSplitterFactory()
    structured_docs = splitter.pre_split_markdown(raw_docs)

    # 5. 存入向量库与 BM25 (Storage 内部会调用 Parent/Child Splitter)
    storage.build_vector_bm25_index(structured_docs)
    
    print("✅ 全部完成！")

if __name__ == "__main__":
    main()