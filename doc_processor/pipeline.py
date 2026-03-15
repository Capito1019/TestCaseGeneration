import os
import json
import shutil  # 新增：用于复制文件
from pathlib import Path

from . import config
from . import doc_to_json
from . import json_process_simplier
from . import json_process_images

# --- 配置 ---
API_TOKEN = config.API_TOKEN
MINERU_BASE_URL = config.MINERU_BASE_URL
# 目标 docs 目录
DOCS_DIR = Path(__file__).parent.parent / "docs" 
# 中间产物总目录
DOC_TO_JSON_OUTPUT_DIR = Path("json_output")

# 状态存储工具函数
def load_status(file_path_dir: Path) -> dict:
    status_path = file_path_dir / "_pipeline_status.json"
    if status_path.exists():
        try:
            return json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_status(file_path_dir: Path, status: dict):
    status_path = file_path_dir / "_pipeline_status.json"
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

# --- 单文件处理逻辑 ---
def process_single_file(local_file_path: Path):
    """
    处理流程优化：
    1. 在 json_output 内部目录完成所有解析和图片转换
    2. 全部成功后，将结果 JSON 同步到 docs 目录
    """
    print(f"\n{'='*50}")
    print(f">>> 开始处理文档: {local_file_path.name}")
    
    # 1. 阶段一：MinerU 解析 (得到如 json_output/文件名/ 目录)
    raw_output_dir_str = doc_to_json.run(
        API_TOKEN, 
        str(local_file_path), 
        MINERU_BASE_URL, 
        str(DOC_TO_JSON_OUTPUT_DIR)
    )
    raw_output_dir = Path(raw_output_dir_str)

    status = load_status(raw_output_dir)
    if not status.get("doc_to_json_done"):
        status["doc_to_json_done"] = True
        save_status(raw_output_dir, status)

    # --- 路径定义：全部设在 raw_output_dir 内部 ---
    raw_layout_json = raw_output_dir / "layout.json"
    # 在 json_output 子目录下先生成中间简化版 JSON
    temp_simplified_json = raw_output_dir / "layout_simplified.json"
    # 最终复制到 docs 的目标路径
    final_dest_json = DOCS_DIR / f"{local_file_path.stem}_simplified.json"

    # 2. 阶段二：结构简化 (在 json_output 内部生成)
    print(f"[Step 2] 正在提取简化信息 -> {temp_simplified_json.name}")
    if status.get("json_simplified"):
        print("  (跳过: 该步骤已完成)")
    else:
        # 注意：此处将 JSON 先保存在中间目录，确保与 images 文件夹同级
        json_process_simplier.run(str(raw_layout_json), str(temp_simplified_json))
        status["json_simplified"] = True
        save_status(raw_output_dir, status)
        print("  (成功: 已生成中间精简版 JSON)")

    # 3. 阶段三：图片处理 (针对中间目录执行)
    print(f"[Step 3] 正在执行图片 OSS 处理 (目录: {raw_output_dir})")
    if status.get("images_processed"):
        print("  (跳过: 该步骤已完成)")
    else:
        # 图片处理脚本现在可以在同级目录下找到 JSON 和 images/ 文件夹
        json_process_images.run(str(raw_output_dir))
        status["images_processed"] = True
        save_status(raw_output_dir, status)
        print("  (成功: 图片 OSS 路径已更新到中间 JSON)")

    # 4. 阶段四：同步结果到 docs 文件夹
    print(f"[Step 4] 正在同步最终结果到 docs 目录...")
    try:
        # 将处理完图片路径的中间 JSON 复制到最终目标位置
        shutil.copy2(temp_simplified_json, final_dest_json)
        print(f"  (完成: 最终 JSON 已就绪 -> {final_dest_json})")
    except Exception as e:
        print(f"  (失败: 复制文件时出错: {e})")

# --- 主入口保持不变 ---
def run_pipeline():
    if not DOCS_DIR.exists():
        print(f"[错误] 目标目录 '{DOCS_DIR}' 不存在，请手动创建。")
        return

    supported_exts = {".docx", ".doc", ".pdf"}
    doc_files = [f for f in DOCS_DIR.iterdir() if f.suffix.lower() in supported_exts]

    if not doc_files:
        print(f"[提示] 在 '{DOCS_DIR}' 目录下未发现待处理文档。")
        return

    print(f"[系统] 扫描完成，共发现 {len(doc_files)} 个文档待解析。")

    for doc_file in doc_files:
        try:
            process_single_file(doc_file)
        except Exception as e:
            print(f"[失败] 处理 {doc_file.name} 时发生错误: {e}")

    print("\n[完成] 所有解析任务执行完毕。")