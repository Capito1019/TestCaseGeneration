import json
import re
from call_llm import get_llm_response
from utils import RequirementEntry
from rag_engine import RAGEngine

class TestGenerator:
    # Markdown Few-shot 示例（基于文档中的“CAN总线溢出监控功能测试”）
    few_shot_example = """
        ### 需求原文 （输入）：
        用户需求ID: UR-R-RB-02
        用户需求内容: 对内外接口均需设计监控措施，使用校验和来检验数据传输的正确性，收到错误的数据请求指令时，能有效屏蔽
        软件需求ID: SR-F-SM-SAFE-03
        软件需求内容: 本功能目的是对CAN的工作状态进行监控
        软件子需求细节: 
        输入 CAN总线无符合协议数据计数
        处理过程 1、各个下位机增加一条数据型间接指令：CAN复位。通过这条指令能够对CAN进行重新初始化。
        2、各下位机增加CAN复位计数遥测。
        3、建议分别独立设置CAN的接收数据区、接收指针及其他总线状态量，避免两条总线之间的互相干扰的可能；
        4、CAN接收缓冲区采取防溢出措施，避免总线异常时，对软件造成灾难性影响；
        5、连续5秒当前总线未收到矢量字查询，进行总线切换，切换总线后连续5秒当前总线仍未收到矢量字查询，进行总线复位；
        6、总线接收、发送错误计数器计数超过90，复位对应CAN总线；
        7、连续200秒总线ES状态错误，复位对应CAN总线；
        8、连续16秒CAN总线中断或离线，复位对应CAN总线。
        输出 CAN重新初始化标记
        CAN上次复位原因
        可靠性优先级 5（1~5，1最低）
        进度优先级 5（1~5，1最低）

        ### 参考示例格式（请严格对齐此结构）：
        # 测试用例报告 1
        - **测试用例名称**: CAN总线溢出监控功能测试
        - **标识**: QC-TC-GN-XTJK-CAN02
        - **追踪关系**: SR-F-SM-SAFE-03 (针对处理过程 4)
        - **测试用例综述**: 检验CAN总线溢出监控功能是否满足需求规定。
        - **用例初始化**: 无
        - **前提和约束**: 无

        #### 测试步骤
        | 序号 | 输入及操作 | 期望结果与评估标准 | 实测结果 |
        | :--- | :--- | :--- | :--- |
        | 1 | 接收机选择A总线通信，修改代码使软件运行一段时间后CAN A接收缓冲区溢出标志InsCanReg[0]置位，在线运行代码，查看状态 | 总线接收缓冲区溢出时，CAN总线重新初始化，CAN总线复位计数加1，复位原因为数据溢出 | |
        | 2 | 接收机选择B总线通信，修改代码使软件运行一段时间后CAN B接收缓冲区溢出标志InsCanReg[1]置位，在线运行代码，查看状态 | 总线接收缓冲区溢出时，CAN总线重新初始化，CAN总线复位计数加1，复位原因为数据溢出 | |

        - **测试用例终止条件**: 本测试用例的全部测试步骤被执行或因某种原因导致测试步骤无法执行。
        - **测试用例通过准则**: 本测试用例的全部测试步骤都通过即标志本用例为“通过”。

        ---

        # 测试用例报告 2
        - **测试用例名称**: CAN总线中断关闭或离线监控功能测试
        - **标识**: QC-TC-GN-XTJK-CAN03
        - **追踪关系**: SR-F-SM-SAFE-03 (针对处理过程 8)
        - **测试用例综述**: 检验CAN总线中断关闭或离线监控功能是否满足需求规定。
        - **用例初始化**: 无
        - **前提和约束**: 无

        #### 测试步骤
        | 序号 | 输入及操作 | 期望结果与评估标准 | 实测结果 |
        | :--- | :--- | :--- | :--- |
        | 1 | 接收机选择A总线通信，修改代码使软件正常运行一段时间后CAN A连续16秒中断位置位关闭，在线运行代码，查看状态 | 总线中断关闭时，CAN总线重新初始化，CAN总线复位计数加1 | |
        | 2 | 修改代码使软件正常运行一段时间后CAN A状态寄存器（SR.Bit7）连续16秒离线状态，在线运行代码，查看状态 | 总线连续16秒离线状态时，CAN总线重新初始化，CAN总线复位计数加1 | |

        - **测试用例终止条件**: 本测试用例的全部测试步骤被执行或因某种原因导致测试步骤无法执行。
        - **测试用例通过准则**: 本测试用例的全部测试步骤都通过即标志本用例为“通过”。
            """
    
    few_shot_planner = """
        ### 示例输入：
        【待处理需求上下文】：
        - 用户需求内容: 系统应支持 CAN 总线冗余切换，主备切换延迟小于 100ms。
        - 软件需求内容: CAN 总线切换控制
        - 软件子需求内容: 1. 软件实时监控 CAN_A 状态寄存器 (0x4000) Bit 0。2. 若 Bit 0 为 1 (离线)，则切换至 CAN_B 片选 (0x4004)。3. 记录切换计数变量 g_switch_cnt。

        ### 示例输出（JSON）：
        [
        {
            "report_index": "1",
            "name": "CAN总线A片选地址读写测试",
            "summary": "验证对 0x74000000 地址的读写操作是否正确触发片选。",
            "suggested_query": "0x74000000 CAN_A_CS 读写操作 寄存器定义"
        },
        {
            "report_index": "2",
            "name": "CAN总线A访问等待周期测试",
            "summary": "验证对 0x74000000 访问时，硬件是否准确执行了 7 个等待周期的延迟逻辑。",
            "suggested_query": "CAN总线 7等待周期 0x74000000 时序要求"
        }
        ]
        """

    def __init__(self, max_rounds=3):
        self.max_rounds = int(max_rounds)
        self.rag = None
        # 初始化 RAG 引擎并加载库

    def _get_technical_context(self, entry: RequirementEntry, search_query: str = None):
        """
        利用 RAG 引擎检索细节。
        修改点：优先使用传入的 search_query（来自规划分支）。
        """
        if not search_query:
            # 兜底逻辑：如果没传 query，使用基础关键词
            search_query = f"{entry.sr_id} {entry.sr_content} 寄存器 宏定义"
        
        context = self.rag.query(search_query, k=3)
        if not context.strip():
            return "（注：未匹配到直接参考，请根据航天软件通用规范生成。）"
        return context

    # --- 第一阶段：构建规划者 Prompt ---
    def build_planner_prompt(self, entry: RequirementEntry):
        sys_prompt = (
            "你是一名资深航天软件测试架构师。你的任务是将复杂的软件需求拆解为若干个独立的测试用例分支。\n"
            "【拆解原则】\n"
            "1. 覆盖全面：必须包含正常功能、边界值、异常保护（如超时、溢出、错误计数）等维度。\n"
            "2. 职责单一：每个分支仅关注一个核心测试点。\n"
            "3. 对于每一个分支，你需要编写一个专门用于检索底层技术细节的 RAG Query，该 Query 应包含需求中的核心关键字、十六进制地址或具体的变量名。"
        )
        user_prompt = (
        f"【示例参考】：\n{self.few_shot_planner}\n\n" # 注入 Few-shot 引导
        f"【待处理需求上下文】：\n"
        f"- 用户需求内容: {entry.ur_content}\n"
        f"- 软件需求内容: {entry.sr_content}\n"
        f"- 软件子需求内容(细节描述): {entry.sr_sub_content}\n\n"
        "【任务要求】：\n"
        "请参考示例，以 JSON 列表格式输出当前需求的测试分支。确保 technical_focus 字段包含具体的地址（如 0x74000000）或变量名。"
    )
        return str(sys_prompt), str(user_prompt)
    
    # --- 执行规划逻辑 ---
    def plan_test_branches(self, entry: RequirementEntry):
        print(f"[Planner] 正在规划测试分支: {entry.sr_id}...")
        sys_p, user_p = self.build_planner_prompt(entry)
        response = get_llm_response(sys_p, user_p)
        
        if response.startswith("ERROR:"):
            return [], {"error": response}

        try:
            # 清理 Markdown 标记并解析 JSON
            json_str = re.sub(r'```json\s*|\s*```', '', response).strip()
            branches = json.loads(json_str)
            return branches, {"sys_p": sys_p, "user_p": user_p, "response": response}
        except Exception as e:
            print(f"[Planner 警告] JSON 解析失败: {e}")
            # 兜底：返回一个基于原始需求的单分支
            fallback = [{"report_index": "1", "name": "全量功能测试", "summary": entry.sr_content, "suggested_query": entry.sr_content}]
            return fallback, {"error": "JSON_PARSE_FAILED", "raw_res": response}
        
    # --- 第二阶段：构建生成器与评审器 Prompt（针对分支优化） ---       
    def build_generator_prompt(self, entry: RequirementEntry, branch: dict, feedback=None):
        """构建生成器 Prompt：注入 RAG 知识背景"""
        tech_context = self._get_technical_context(entry, branch.get("suggested_query"))

        """
        优化后的生成器 Prompt 对。
        1. 增加 Markdown Few-shot 示例 。
        2. 严格规定输出格式，禁止多余解释。
        """
        # 系统提示词：定义角色与严苛的输出规则
        sys_prompt = (
            "你是一名资深嵌入式测试工程师。请根据用户提供的软件需求编写测试用例。\n"
            "【强制规则】\n"
            f"1. 必须且只能生成“一个”测试用例报告，标题必须严格命名为：# 测试用例报告 {branch['report_index']}\n"
            "2. 必须严格遵守提供的 Markdown 格式示例，包含所有字段和表格结构。\n"
            "3. 输出内容必须直接是测试用例，禁止包含任何引导语、解释性文字、开场白或结束语。\n"
            "4. 针对特定的【测试分支目标】编写精细化的测试用例步骤。"
            "5.必须优先使用【底层技术参考资料】中提供的真实寄存器、变量和宏定义。如果资料中提及了具体的 Bit 位或偏移地址，必须在测试步骤中体现。\n"
        )
    

        # 用户提示词：注入具体需求与反馈
        feedback_info = f"\n\n【评审意见反馈 - 请根据以下意见优化完善用例】\n{feedback}" if feedback else "\n\n请直接输出首版测试用例。"
        
        user_prompt = (
            f"【测试分支目标】：\n- 名称: {branch['name']}\n- 综述: {branch['summary']}\n"
            f"【底层技术参考资料】：\n{tech_context}\n\n" # 注入检索到的真实细节
            f"【测试用例示例】:{self.few_shot_example}\n"
            f"---\n"
            f"【待处理需求上下文】\n"
            f"- 用户需求ID: {entry.ur_id} \n"
            f"- 用户需求内容: {entry.ur_content} \n"
            f"- 软件需求ID: {entry.sr_id} \n"
            f"- 软件需求内容: {entry.sr_content} \n"
            f"- 软件子需求细节: {entry.sr_sub_content} \n"
            f"{feedback_info}"
        )
        
        return str(sys_prompt), str(user_prompt)
    
     # --- 第三阶段：评审阶段 ---  
    def build_reviewer_prompt(self, entry: RequirementEntry, branch: dict, generated_case: str):
        """构建评审器 Prompt：利用 RAG 进行真实性校验"""
        # 获取该分支相关的技术背景
        tech_context = self._get_technical_context(entry, branch.get("suggested_query"))
        """
        优化后的评审器 Prompt 对。
        1. 强化对“白盒故障注入”手段的审查 。
        2. 严格检查 Markdown 格式及追踪关系 。
        3. 增加对输入操作清晰度（如寄存器/变量名）的校验 。
        """
        # 系统提示词：定义评审的高级准则与“挑剔”的专家角色
        sys_prompt = (
            "你是一名资深航天软件测试架构师，负责对测试用例进行最终质量把关。\n"
            "你的评审标准极其严苛，必须针对以下维度进行逐项核对：\n"
            "1. 分支达成度：用例是否完全实现了【当前测试分支目标】所描述的逻辑？\n"
            "2. 测试手段：对于溢出、离线、中断等异常逻辑，是否采用了“修改代码/置位寄存器标志位”等白盒测试方法？（拒绝仅描述“模拟”而无具体操作的模糊描述）。\n"
            "3. 步骤严谨性：输入操作是否指明了具体的变量名、寄存器或地检指令？期望结果是否对应了具体的遥测参数或复位计数变化？。\n"
            "4. 格式规范：标识是否符合 QC-TC-GN 规范？追踪关系是否准确包含了对应的 UR 和 SR ID？Markdown 结构是否与示例完全一致？。"
        )
        
        # 用户提示词：提供比对素材与明确的任务指令
        user_prompt = (
            f"【测试分支目标】：\n- 名称: {branch['name']}\n- 综述: {branch['summary']}\n\n"
            f"【底层技术参考资料】：\n{tech_context}\n\n" # 注入检索到的真实细节
            f"-测试用例示例:{self.few_shot_example}\n"
            f"---\n"
            f"### 原始软件需求上下文\n"
            f"- 用户需求ID: {entry.ur_id} \n"
            f"- 用户需求内容: {entry.ur_content} \n"
            f"- 软件需求ID: {entry.sr_id} \n"
            f"- 软件需求内容: {entry.sr_content} \n"
            f"- 软件子需求细节: {entry.sr_sub_content} \n"
            f"---\n"
            f"### 待评审测试用例（Markdown）\n"
            f"{generated_case}\n\n"
            f"--- \n"
            f"【评审任务】\n"
            f"先根据所给的测试用例示例补充评审标准，然后请对比需求原文与生成的用例，并进行打分：\n"
            f"1. 如果该用例已达到“可以直接用于航天软件测试实战”的精细度且无任何遗漏，请只回复“PASS”。\n"
            f"2. 如果发现任何：逻辑遗漏、操作描述模糊（未提及具体寄存器/变量）、格式不符、或未采用白盒注入手段的情况，请指出具体的问题点。\n"
            f"【注意】评审意见必须简练、直接，直接指出需要修改的步骤序号或字段。"
        )
        
        return str(sys_prompt), str(user_prompt)

# --- 迭代闭环逻辑 ---
    def generate_case_for_branch(self, entry: RequirementEntry, branch: dict):
        current_case, feedback = "", None
        history = []
        for r in range(self.max_rounds):
            round_data = {"round": r + 1}
            
            # 1. 生成环节
            sg, ug = self.build_generator_prompt(entry, branch, feedback)
            current_case = get_llm_response(sg, ug)
            if current_case.startswith("ERROR:"): break
            
            # 记录生成 Prompt
            round_data["generator_prompts"] = {"system": sg, "user": ug}
            round_data["gen_res"] = current_case

            # 2. 评审环节
            sr, ur = self.build_reviewer_prompt(entry, branch, current_case)
            feedback = get_llm_response(sr, ur)
            if feedback.startswith("ERROR:"): break
            
            # 记录评审 Prompt
            round_data["reviewer_prompts"] = {"system": sr, "user": ur}
            round_data["rev_res"] = feedback
            
            history.append(round_data)
            if "PASS" in feedback.upper() and len(feedback) < 10: break
            
        return current_case, history

    # --- 总调度 ---
    def generate_refined_case(self, entry: RequirementEntry):
        branches, plan_log = self.plan_test_branches(entry)
        all_cases, full_history = [], {"plan_log": plan_log, "branch_results": []}
        
        # 使用 enumerate 进行自动编号
        for i, branch in enumerate(branches, 1):
            print(f"  [Branch] 正在处理分支 {i}: {branch['name']}")
            
            # 将编号注入 branch 字典供生成器使用
            branch["report_index"] = i
            
            case_text, branch_history = self.generate_case_for_branch(entry, branch)
            all_cases.append(case_text)
            
            # 在 JSON 结构中显式标注报告编号
            full_history["branch_results"].append({
                "report_id": f"测试用例报告 {i}",
                "branch_info": branch,
                "history": branch_history
            })
            
        return "\n\n---\n\n".join(all_cases), full_history