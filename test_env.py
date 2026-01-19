import sys
import os

# 打印出 Python 到底在哪些地方找包
print("--- 搜索路径 ---")
for p in sys.path:
    print(p)

try:
    from langchain_community.retrievers.ensemble import EnsembleRetriever
    print("\n✅ 物理文件校验成功！EnsembleRetriever 位置:", EnsembleRetriever.__module__)
except Exception as e:
    print(f"\n❌ 依然失败: {e}")