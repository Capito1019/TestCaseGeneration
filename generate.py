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

    def __init__(self, max_rounds=3):
        self.max_rounds = int(max_rounds)
        # 初始化 RAG 引擎并加载库

    def _get_technical_context(self, entry: RequirementEntry):
        """利用 RAG 引擎检索与需求高度相关的底层细节"""
        # 构造混合查询语句：结合 SR ID 和具体业务关键词
        search_query = f"{entry.sr_id} {entry.sr_content} 寄存器 宏定义 变量名 状态位"
        # 混合召回 + 重排序获取 Top 3 片段
        context = self.rag.query(search_query, k=3)
        
        if not context.strip():
            return "（注：当前需求在知识库中未匹配到直接的寄存器或宏定义参考，请根据航天软件通用规范进行生成。）"
        return context

    def build_generator_prompt(self, entry: RequirementEntry, feedback=None):
        """构建生成器 Prompt：注入 RAG 知识背景"""
        tech_context = self._get_technical_context(entry)
        """
        优化后的生成器 Prompt 对。
        1. 增加 Markdown Few-shot 示例 。
        2. 严格规定输出格式，禁止多余解释。
        """
        # 系统提示词：定义角色与严苛的输出规则
        sys_prompt = (
            "你是一名资深嵌入式测试工程师。请根据用户提供的软件需求编写测试用例。\n"
            "【强制规则】\n"
            "1. 必须严格遵守提供的 Markdown 格式示例，包含所有字段和表格结构。\n"
            "2. 输出内容必须直接是测试用例，禁止包含任何引导语、解释性文字、开场白或结束语。\n"
            "3. 覆盖要求：必须涵盖需求中所有的处理过程分支，包括正常逻辑与异常处理。"
            "4. 每个用例必须拥有唯一的标识（如 QC-TC-GN-001, 002...）。"
            "5. 对输出的测试用例进行细粒度切分（3-6个），确保每个子测试用例的职责细分"
            "6.必须优先使用【底层技术参考资料】中提供的真实寄存器、变量和宏定义。如果资料中提及了具体的 Bit 位或偏移地址，必须在测试步骤中体现。\n"
        )
    

        # 用户提示词：注入具体需求与反馈
        feedback_info = f"\n\n【评审意见反馈 - 请根据以下意见优化完善用例】\n{feedback}" if feedback else "\n\n请直接输出首版测试用例。"
        
        user_prompt = (
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
        
        return sys_prompt, user_prompt

    def build_reviewer_prompt(self, entry: RequirementEntry, generated_case: str):
        """构建评审器 Prompt：利用 RAG 进行真实性校验"""
        tech_context = self._get_technical_context(entry)
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
            "1. 逻辑覆盖：是否完全覆盖了需求中的所有处理分支（如 5s/16s/200s 超时、错误计数>90、校验错误等）。\n"
            "2. 测试手段：对于溢出、离线、中断等异常逻辑，是否采用了“修改代码/置位寄存器标志位”等白盒测试方法？（拒绝仅描述“模拟”而无具体操作的模糊描述）。\n"
            "3. 步骤严谨性：输入操作是否指明了具体的变量名、寄存器或地检指令？期望结果是否对应了具体的遥测参数或复位计数变化？。\n"
            "4. 格式规范：标识是否符合 QC-TC-GN 规范？追踪关系是否准确包含了对应的 UR 和 SR ID？Markdown 结构是否与示例完全一致？。"
            "5. 用例粒度：生成的测试用例需控制在3~6个，确保每个子测试用例的职责细分"
        )
        
        # 用户提示词：提供比对素材与明确的任务指令
        user_prompt = (
            f"【底层技术参考资料】：\n{tech_context}\n\n" # 注入检索到的真实细节
            f"-测试用例示例:{self.few_shot_example}\n"
            f"---\n"
            f"### 原始软件需求原文\n"
            f"```\n{entry.sr_sub_content}\n```\n\n"
            f"### 待评审测试用例（Markdown）\n"
            f"{generated_case}\n\n"
            f"--- \n"
            f"【评审任务】\n"
            f"先根据所给的测试用例示例补充评审标准，然后请对比需求原文与生成的用例，并进行打分：\n"
            f"1. 如果该用例已达到“可以直接用于航天软件测试实战”的精细度且无任何遗漏，请只回复“PASS”。\n"
            f"2. 如果发现任何：逻辑遗漏、操作描述模糊（未提及具体寄存器/变量）、格式不符、或未采用白盒注入手段的情况，请指出具体的问题点。\n"
            f"【注意】评审意见必须简练、直接，直接指出需要修改的步骤序号或字段。"
        )
        
        return sys_prompt, user_prompt

    def generate_refined_case(self, entry: RequirementEntry):
        """
        执行“生成-评审-完善”循环，并记录全过程。
        增加了针对网络错误的断路保护逻辑。
        """
        current_case = ""
        feedback = None
        history = []

        for r in range(self.max_rounds):
            round_data = {"round": r + 1, "generator": {}, "reviewer": {}}
            
            # --- 1. 生成者环节 ---
            sys_gen, user_gen = self.build_generator_prompt(entry, feedback)
            current_case = get_llm_response(sys_gen, user_gen)
            
            # 【关键修改】：检查生成器是否报错
            if current_case.startswith("ERROR:"):
                print(f"[跳过] 需求 {entry.sr_id} 处理失败：生成阶段发生网络异常。")
                round_data["generator"] = {"error": current_case}
                history.append(round_data)
                break # 直接终止当前需求的迭代循环

            round_data["generator"] = {
                "system_prompt": sys_gen,
                "user_prompt": user_gen,
                "response": current_case
            }

            # --- 2. 评审者环节 ---
            sys_rev, user_rev = self.build_reviewer_prompt(entry, current_case)
            feedback = get_llm_response(sys_rev, user_rev)
            
            # 【关键修改】：检查评审器是否报错
            if feedback.startswith("ERROR:"):
                print(f"[警告] 需求 {entry.sr_id} 评审中断：评审阶段发生网络异常。")
                round_data["reviewer"] = {"error": feedback}
                history.append(round_data)
                break # 终止循环，防止带入错误的反馈进入下一轮

            round_data["reviewer"] = {
                "system_prompt": sys_rev,
                "user_prompt": user_rev,
                "response": feedback
            }

            history.append(round_data)
            
            # 正常判定是否通过
            if "PASS" in feedback.upper() and len(feedback) < 10:
                print(f"[完成] {entry.sr_id} 评审通过，迭代轮次: {r+1}")
                break
        
        return current_case, history