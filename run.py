import argparse

from utils import RequirementProcessor, RequirementEntry
from generate import TestGenerator
from rag_engine import RAGEngine
from pathlib import Path

import os
import json
import glob

from doc_processor.pipeline import run_pipeline as start_parsing


def process_and_save(entry: RequirementEntry, generator: TestGenerator):
    """
    调用迭代生成逻辑，并处理结果输出
    """
    # 执行“生成-评审-完善”循环 [cite: 10]
    final_test_case = generator.generate_refined_case(entry)
    
    print(f"\n{'='*20} 最终测试用例结果 {'='*20}")
    print(final_test_case)
    print(f"{'='*57}\n")
    
    # 建议：此处可以增加将结果保存为 .md 或 .docx 文件的逻辑
    # save_to_file(entry.sr_id, final_test_case)

def manage_knowledge_base(rag_engine, docs_folder="docs", index_path="faiss_index"):
    """自动化管理知识库：包含文档解析与知识库重建"""
    config_file = os.path.join(index_path, "kb_config.json")
    
    # 这里的指纹会扫描 docs 文件夹下所有文件（包括 docx 和 json）
    # 只要原始 docx 变动，指纹就会变，从而触发 need_rebuild
    current_fingerprint = rag_engine.get_docs_fingerprint(docs_folder)
    
    need_rebuild = False
    if not os.path.exists(index_path) or not os.path.exists(config_file):
        print("[RAG] 未发现知识库或配置文件，准备首次建立...")
        need_rebuild = True
    else:
        with open(config_file, "r") as f:
            old_config = json.load(f)
        if old_config.get("fingerprint") != current_fingerprint:
            print("[RAG] 检测到文档更新（或中间产物变化），准备重构知识库...")
            need_rebuild = True

    if need_rebuild:
        if start_parsing:
            print("[系统] 启动文档解析流水线 (MinerU + Simplified Logic)...")
            start_parsing() # 执行解析逻辑
        else:
            print("[错误] 解析函数不可用，请检查 doc_processor 目录结构。")
            return

        # 经过解析后，docs 目录下应已生成了对应的 *_simplified.json 文件
        json_files = glob.glob(os.path.join(docs_folder, "*_simplified.json"))
        
        if not json_files:
            print(f"[RAG] 警告：在 {docs_folder} 目录下未找到解析生成的 JSON 文件。")
            return
        
        print(f"[RAG] 正在基于 {len(json_files)} 个结构化文档构建知识库...")
        # 此时调用的是你之前优化过的支持层级划分的 build_knowledge_base
        rag_engine.build_knowledge_base(json_files)
        
        # 保存新指纹
        os.makedirs(index_path, exist_ok=True)
        with open(config_file, "w") as f:
            json.dump({"fingerprint": current_fingerprint}, f)
        print("[RAG] 知识库已更新并同步指纹。")
    else:
        print("[RAG] 知识库已是最新状态。")
        rag_engine.load_db()

def main():
    # 1. 初始化参数解析器
    parser = argparse.ArgumentParser(description="浙大AI接口测试用例生成 Pipeline 入口")

    # 2. 定义 7 个核心参数
    parser.add_argument("--excel_path", type=str, help="Excel/CSV文件路径")
    parser.add_argument("--ur_id", type=str, help="用户需求ID (如 UR-IO-NB-01)")
    parser.add_argument("--ur", type=str, help="用户需求内容")
    parser.add_argument("--sr_type", type=str, help="软件需求类型 (如 功能/性能)")
    parser.add_argument("--sr_id", type=str, help="软件需求ID (如 SR-F-DEM-01)")
    parser.add_argument("--sr", type=str, help="软件需求内容")
    parser.add_argument("--sr_sub", type=str, help="软件子需求内容")
    parser.add_argument("--max_rounds", type=str, default="3", help="迭代轮次")
    parser.add_argument("--output", type=str, default="results.xlsx", help="输出Excel文件名")

    args = parser.parse_args()

# --- 【关键修改点】：在初始化生成器前，统一管理知识库 ---
    rag_engine = RAGEngine()
    manage_knowledge_base(rag_engine)
    
    # 将已管理好的 rag 引擎传入生成器
    max_rounds = args.max_rounds if args.max_rounds is not None else 3
    generator = TestGenerator(max_rounds= max_rounds)
    generator.rag = rag_engine 

    # 后续处理逻辑保持不变
    entries = []
    if args.excel_path:
        entries = RequirementProcessor.process_excel(args.excel_path)
    elif all([args.ur_id, args.ur, args.sr_type, args.sr_id, args.sr, args.sr_sub]):
        entries = [RequirementEntry(args.ur_id, args.ur, args.sr_type, args.sr_id, args.sr, args.sr_sub)]

    if not entries:
        print("提示：未提供有效输入。")
        return

    for entry in entries:
        print(f"\n>>> 正在处理需求: {entry.sr_id}")
        final_case, history = generator.generate_refined_case(entry)
        
        RequirementProcessor.save_individual_case(entry, final_case, history, "test_results")
        RequirementProcessor.save_as_markdown(entry, final_case, "test_results")

if __name__ == "__main__":
    main()