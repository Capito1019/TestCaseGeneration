import pandas as pd
from dataclasses import dataclass
from typing import List, Optional
import os
import json

@dataclass
class RequirementEntry:
    """存储需求对应关系的数据对象"""
    ur_id: str             # 用户需求ID
    ur_content: str        # 用户需求内容
    sr_type: str           # 软件需求类型
    sr_id: str             # 软件需求ID
    sr_content: str        # 软件需求内容
    sr_sub_content: str    # 软件子需求内容 (核心生成点)

class RequirementProcessor:
    @staticmethod
    def process_excel(file_path: str) -> List[RequirementEntry]:
        """
        处理Excel/CSV文件，确保行列对应关系，并返回对象列表。
        """
        # 读取数据
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        # 核心逻辑：向下填充 (Forward Fill)
        # 解决Excel合并单元格在读取时只有第一行有值，后续行为NaN的问题 
        columns_to_fill = ['用户需求ID', '用户需求', '软件需求类型', '软件需求ID', '软件需求']
        df[columns_to_fill] = df[columns_to_fill].ffill()

        # 过滤掉“软件子需求”为空的行，确保每一条子需求都有对应数据
        df = df.dropna(subset=['软件子需求'])

        entries = []
        for _, row in df.iterrows():
            entry = RequirementEntry(
                ur_id=str(row.get('用户需求ID', '')),
                ur_content=str(row.get('用户需求', '')),
                sr_type=str(row.get('软件需求类型', '')),
                sr_id=str(row.get('软件需求ID', '')),
                sr_content=str(row.get('软件需求', '')),
                sr_sub_content=str(row.get('软件子需求', ''))
            )
            entries.append(entry)
        
        return entries
    
    @staticmethod
    def save_individual_case(entry, final_case, history, base_dir="test_results"):
        """
        保存单个需求的完整日志，包含每一轮的 Prompt 和 Response
        """
        os.makedirs(base_dir, exist_ok=True)
        
        # 构造完整的数据结构
        result_data = {
            "metadata": {
                "ur_id": entry.ur_id,
                "ur_content": entry.ur_content,
                "sr_type": entry.sr_type,           
                "sr_id": entry.sr_id,
                "sr_content": entry.sr_content,
                "sr_sub_content": entry.sr_sub_content
            },
            "final_test_case": final_case,
            "iteration_history": history  # 这里包含了每一轮的完整 Prompt
        }
        
        file_path = os.path.join(base_dir, f"{entry.sr_id}_result.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        print(f"[存储] 详细日志已保存至: {file_path}")

    @staticmethod
    def save_as_markdown(entry, final_case, base_dir="test_results"):
        """
        将生成的测试用例保存为 Markdown 格式，增强可读性。
        """
        # 1. 创建存储目录
        safe_sr_id = entry.sr_id.replace(":", "_").replace("/", "_")
        case_dir = os.path.join(base_dir, safe_sr_id)
        os.makedirs(case_dir, exist_ok=True)

        # 2. 构造 Markdown 内容
        md_content = [
            f"# 测试用例生成报告 - {entry.sr_id}\n",
            "## 1. 需求追溯上下文",
            f"- **用户需求ID**: `{entry.ur_id}`",
            f"- **软件需求ID**: `{entry.sr_id}`",
            f"- **需求类型**: {entry.sr_type}",
            "\n### 原始软件子需求细节",
            f"> {entry.sr_sub_content}\n",
            "---\n",
            "## 2. 生成的测试用例内容\n",
            final_case,  # 直接注入大模型输出的内容
            "\n---\n",
            f"> *生成日期: 2026-01-28*"
        ]

        # 3. 写入文件
        md_path = os.path.join(case_dir, f"{safe_sr_id}_case_report.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_content))
        
        print(f"[系统] Markdown 报告已生成: {md_path}")