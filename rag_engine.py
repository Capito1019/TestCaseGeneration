import os
import json
import re
import hashlib
import glob
from pathlib import Path
from sentence_transformers import CrossEncoder

# 核心库引用
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

class RAGEngine:
    def __init__(self, db_path="faiss_index"):
        self.db_path = db_path
        # 稠密向量模型 (Bi-Encoder)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # 重排序模型 (Cross-Encoder)
        self.reranker = CrossEncoder('BAAI/bge-reranker-base', device='cpu') 
        self.vector_db = None
        self.bm25_retriever = None # 稀疏检索器

    def _get_title_level(self, title_text):
        """识别标题层级"""
        clean_text = title_text.strip().rstrip('.')
        match = re.match(r'^(\d+(\.\d+)*)', clean_text)
        if match:
            return len(match.group(1).split('.'))
        return 1

    def _is_noise_block(self, text):
        text = text.strip()
        if len(text) < 2: return True
        
        # 1. 基础目录点过滤 (覆盖 1引言... 等)
        if "..." in text or "……" in text or ".." in text:
            return True
            
        # 2. 增强型“页码”指纹过滤 (覆盖 结尾是 .16, )9, -20 等情况)
        # 逻辑：匹配 [点号/中英文右括号/空格/横杠] 后跟 1-3 位数字结尾
        if re.search(r'(\.|\s|…|—|）|\)|-)\d{1,3}$', text):
            return True

        # 3. 关键词黑名单 (覆盖 目录、图表目录 标题)
        if text in ["目录", "目 录", "图表目录", "图 录", "文档控制", "变更记录"]:
            return True

        # 4. 针对特殊碎片：如 "会签"、"页" 等无意义短文本
        if text in ["会签", "会 签"] or re.match(r'^第\s*\d+\s*页', text):
            return True

        return False

    def _process_json_file(self, file_path):
        """针对 simplified.json 的专业分块与层级路径注入"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        all_docs = []
        title_stack = []
        current_content = []
        current_text_len = 0
        has_media = False

        blocks = []
        for page in data.get("pdf_info", []):
            blocks.extend(page.get("para_blocks", []))

        def finalize_chunk():
            nonlocal current_content, current_text_len, has_media
            if current_text_len > 20 or has_media:
                path_str = " > ".join(title_stack)
                # 注入标题路径以增强语义和关键词命中
                page_content = f"【标题路径】：{path_str}\n\n" + "\n".join(current_content)
                
                doc = Document(
                    page_content=page_content,
                    metadata={
                        "source": os.path.basename(file_path),
                        "title_path": list(title_stack),
                        "full_path": path_str
                    }
                )
                all_docs.append(doc)
            current_content = []
            current_text_len = 0
            has_media = False

        for block in blocks:
            b_type = block.get("type")
            text = block.get("text", "").strip()

            if b_type == "title":
                # 如果标题被判定为目录噪音，直接跳过，不让它污染 title_stack
                if self._is_noise_block(text):
                    continue
                if current_content: 
                    finalize_chunk()
                level = self._get_title_level(text)
                while len(title_stack) >= level:
                    title_stack.pop()
                title_stack.append(text)
            elif b_type in ["text", "table", "image"]:
                if b_type == "text":
                    txt = block.get("text", "").strip()
                    if self._is_noise_block(txt): continue
                    current_content.append(txt)
                    current_text_len += len(txt)
                elif b_type == "table":
                    current_content.append(block.get("html", ""))
                    has_media = True
                elif b_type == "image":
                    paths = ", ".join(block.get("image_paths", []))
                    current_content.append(f"[图片路径]: {paths}")
                    has_media = True

        if current_content: finalize_chunk()
        return all_docs

    def build_knowledge_base(self, file_paths: list):
        """建立包含 FAISS 和 BM25 的混合知识库"""
        all_docs = []
        for path in file_paths:
            if path.endswith(".json"):
                all_docs.extend(self._process_json_file(path))
            else:
                loader = TextLoader(path, encoding="utf-8") if path.endswith(".txt") else UnstructuredMarkdownLoader(path)
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                all_docs.extend(text_splitter.split_documents(documents))

        if not all_docs:
            print("[RAG] 警告：无有效内容")
            return

        # 1. 存储稠密向量库
        self.vector_db = FAISS.from_documents(all_docs, self.embeddings)
        self.vector_db.save_local(self.db_path)

        # 2. 初始化 BM25 检索器
        self.bm25_retriever = BM25Retriever.from_documents(all_docs)
        print(f"[RAG] 混合知识库已建立。共存储 {len(all_docs)} 个逻辑区块。")

    def load_db(self):
        """加载向量库并重建 BM25 检索器"""
        if os.path.exists(self.db_path):
            self.vector_db = FAISS.load_local(self.db_path, self.embeddings, allow_dangerous_deserialization=True)
            # 从 FAISS 的 docstore 中提取所有文档以重建 BM25
            all_docs = list(self.vector_db.docstore._dict.values())
            self.bm25_retriever = BM25Retriever.from_documents(all_docs)
            print("[RAG] 成功加载本地向量库并重建 BM25 检索器")
        else:
            print("[RAG] 警告：未找到本地知识库索引")

    def query(self, question: str, k=3, top_n=15):
        """两阶段检索：混合召回 (BM25 + FAISS) + 重排序 (Rerank)"""
        if not self.vector_db or not self.bm25_retriever:
            print("[RAG] 检索器未初始化")
            return ""

        # 第一阶段：混合召回
        faiss_retriever = self.vector_db.as_retriever(search_kwargs={"k": top_n})
        self.bm25_retriever.k = top_n
        
        # 使用 EnsembleRetriever 进行加权混合打分
        ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, faiss_retriever],
            weights=[0.4, 0.6] # 稠密向量权重略高，处理语义；关键词权重处理术语
        )
        
        initial_docs = ensemble_retriever.get_relevant_documents(question)
        if not initial_docs: return ""

        # 第二阶段：重排序
        pairs = [[question, doc.page_content] for doc in initial_docs]
        scores = self.reranker.predict(pairs)
        doc_scores = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
        
        # 选出最终 Top-K
        final_docs = [doc for doc, score in doc_scores[:k]]
        return "\n".join([doc.page_content for doc in final_docs])

    def get_docs_fingerprint(self, folder_path="docs"):
        """
        修改点：只计算原始源文件的指纹，忽略生成的 _simplified.json
        """
        # 定义你认为的“源文件”格式
        source_exts = {".docx", ".doc", ".pdf", ".txt"}
        
        # 仅获取源文件列表
        files = [
            f for f in glob.glob(os.path.join(folder_path, "*")) 
            if Path(f).suffix.lower() in source_exts
        ]
        
        if not files:
            return None
        
        # 排序以保证计算一致性
        files.sort()
        
        fingerprint_content = ""
        for f in files:
            # 拼接文件名和最后修改时间
            fingerprint_content += f"{f}_{os.path.getmtime(f)}"
        
        return hashlib.md5(fingerprint_content.encode()).hexdigest()

    def inspect_knowledge_base(self):
        """查看知识库状态预览"""
        if not self.vector_db: return
        doc_ids = list(self.vector_db.docstore._dict.keys())
        print(f"\n预览总数: {len(doc_ids)}")
        for i, doc_id in enumerate(doc_ids[:5]): # 仅预览前5个
            doc = self.vector_db.docstore.search(doc_id)
            print(f"【区块 {i+1}】 {doc.metadata.get('full_path')}\n内容预览: {doc.page_content[:150]}...\n{'-'*40}")

# # --- 测试执行块 ---
# if __name__ == "__main__":
#     # 1. 准备测试环境
#     test_rag = RAGEngine(db_path="test_faiss_index")
    
#     # 2. 指定您解析出的简化 JSON 文件路径
#     # 请确保该文件已存在于当前目录或指定路径
#     test_json_path = "docs/定轨接口软件需求规格说明V1.00（公开）_simplified.json"
    
#     if os.path.exists(test_json_path):
#         print(f"[测试] 正在处理文件: {test_json_path}")
        
#         # 3. 建立知识库：执行层级切分与实质性内容过滤
#         test_rag.build_knowledge_base([test_json_path])
        
#         # 4. 调用预览方法检查结果
#         test_rag.inspect_knowledge_base()
        
#         # 5. 可选：测试一条简单的检索，验证标题路径是否增强了语义匹配
#         print("\n[测试] 尝试检索关键接口定义...")
#         res = test_rag.query("CAN接口")
#         print(f"检索结果:\n{res}")
#     else:
#         print(f"[错误] 未找到测试文件: {test_json_path}，请先运行 parser 脚本生成该文件。")