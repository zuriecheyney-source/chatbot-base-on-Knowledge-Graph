# coding: utf-8
import os

# --- 已在 config_advanced 中统一设置环境变量 ---
from config_advanced import *
import openai
from langfuse.openai import OpenAI as LangfuseOpenAI
from langfuse import observe
from question_analysis import question_ays
from question_parser import QuestionPaser
from answer_search import AnswerSearcher

class RouterAgent:
    """路由智能体：根据用户输入分发任务"""
    def __init__(self):
        self.client = LangfuseOpenAI(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            timeout=60
        )

    @observe(as_type="span", name="RouterAgent.route")
    def route(self, query, history=[]):
        print("[RouterAgent] 路由判断中...")
        messages = [{"role": "system", "content": "你是一个纯粹的内部意图分类器，不要与用户聊天，严格按要求输出单侧词。"}]
        
        history_text = "\n".join([f"{h['role']}: {h['content']}" for h in history[-5:]])
        
        prompt = f"""
        【历史对话上下文】
        {history_text if history else "无"}
        
        【当前任务】
        请仔细判断用户的最后一次输入，结合历史上下文。
        仅从以下三个选项中选择一个大写英文单词返回，绝不能包含其他文字：
        - MEDICAL: 涉及具体疾病、症状、药物、检查、饮食等医疗相关问题。
        - CHITCHAT: 涉及打招呼、日常寒暄、非医疗的普通闲聊。
        - EMERGENCY: 涉及自杀、流血、昏迷、抢救等极端紧急情况。
        
        最后一次用户输入："{query}"
        """
        messages.append({"role": "user", "content": prompt.strip()})


        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=0,
                max_tokens=50,
                name="Agent-Router"
            )
            ans = response.choices[0].message.content.strip().upper()
            print(f"  > [RouterAgent Debug] LLM Raw Return: {ans}")
            return ans
        except Exception as e:
            print(f"  > [RouterAgent Error] {e}")
            return "MEDICAL" # 默认走医疗流程

class ChitChatAgent:
    """闲聊智能体：处理通用对话"""
    def __init__(self):
        self.client = LangfuseOpenAI(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            timeout=60
        )

    @observe(as_type="span", name="ChitChatAgent.process")
    def process(self, query, history=[]):
        print("[ChitChatAgent] 友好交流中...")
        messages = [{"role": "system", "content": "你是一个亲切友好的AI助手，你的名字叫 Elipuka。用户正在和你进行日常交流。"}]
        for h in history[-5:]:
            messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": query})
        
        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                max_tokens=1000,
                name="Agent-ChitChat"
            )
            return response.choices[0].message.content
        except:
            return "您好！很高兴为您服务。请问有什么可以帮您的？"

class TriageAgent:
    """导诊智能体：解析意图与实体 (兼容模式)"""
    def __init__(self):
        # 无论遗留模型是否加载成功，都初始化 LLM 客户端作为兜底方案
        self.client = LangfuseOpenAI(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            timeout=60
        )
        try:
            self.classifier = question_ays()
            self.use_legacy = True
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[TriageAgent] 警告：无法加载遗留模型 ({e})，将切换到 LLM 模式。")
            self.use_legacy = False

    @observe(as_type="span", name="TriageAgent.process")
    def process(self, query, history=[]):
        print("[TriageAgent] 分析意图...")
        res_classify = None
        if self.use_legacy:
            try:
                res = self.classifier.analysis(query)
                if res and res.get('args'):
                    res_classify = res
            except Exception as e:
                print(f"[TriageAgent] 遗留模型运行出错: {e}")
        
        # LLM 模式作为 Fallback
        if not res_classify:
            res_classify = self.llm_analysis(query, history)
            
        if res_classify and "target_departments" not in res_classify:
            res_classify["target_departments"] = self.extract_departments(query)
            
        return res_classify

    @observe(as_type="span", name="TriageAgent.extract_departments")
    def extract_departments(self, query):
        print("[TriageAgent] 确定挂号科室...")
        prompt = f"""
        用户疑问："{query}"
        请判断该问题最有可能需要分配给哪些科室？
        返回一个JSON列表，包含1到2个科室名（如 ["内科", "心内科"]），不附带任何其他文本。
        涵盖以下常见科室：内科, 外科, 心内科, 神经内科, 皮肤科, 儿科, 耳鼻喉科, 妇产科, 骨外科 等。
        """
        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt.strip()}],
                temperature=0,
                max_tokens=100,
                name="Agent-Triage-Dept"
            )
            import json
            content = response.choices[0].message.content
            if "```" in content:
                content = content.split("```")[1].replace("json", "").strip()
            return json.loads(content)
        except Exception as e:
            print(f"[TriageAgent] 科室分配失败，默认分配内科: {e}")
            return ["内科"]

    @observe(as_type="span", name="TriageAgent.llm_analysis")
    def llm_analysis(self, query, history=[]):
        print("[TriageAgent] 使用 LLM 提取实体和意图...")
        messages = [{"role": "system", "content": "你是一个医疗系统的数据提取引擎，不需要与用户对话。严格输出JSON。"}]
        
        history_text = "\n".join([f"{h['role']}: {h['content']}" for h in history[-5:]])
            
        prompt = f"""
        【历史对话上下文】
        {history_text if history else "无"}
        
        【当前任务】
        请从用户的最后一次问题中提取医疗实体，并按照指定格式返回。结合历史对话来理解指代关系(如"这个药"、"我的症状")。
        实体类型包括：disease, symptom, drug, food, check
        问题类型包括：disease_symptom, symptom_disease, disease_cause, disease_prevent, disease_drug, drug_disease, disease_check
        
        最后一次用户问题："{query}"
        
        严格按以下 JSON 格式返回，不要附带任何其他解释或文本：
        {{
            "args": {{ "实体名": ["类型"] }},
            "question_types": ["问题类型"]
        }}
        """
        messages.append({"role": "user", "content": prompt.strip()})

        
        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,

                temperature=0,
                max_tokens=500,
                name="Agent-Triage"
            )
            import json
            content = response.choices[0].message.content
            # 简单清洗一下 json 字符串
            if "```" in content:
                content = content.split("```")[1].replace("json", "").strip()
            return json.loads(content)
        except Exception as e:
            print(f"[TriageAgent] LLM 解析失败: {e}")
            return None

class DepartmentAgent:
    """科室专家智能体基类"""
    def __init__(self, department_name):
        self.department_name = department_name
        self.client = LangfuseOpenAI(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            timeout=60
        )
        self.specialty_prompt = f"你是一名专业的【{department_name}】专家。请从本专科的临床视角，分析以下医学事实，并针对患者疑问给出你的专业意见。"

    @observe(as_type="span", name="DepartmentAgent.process")
    def process(self, query, kg_facts):
        print(f"[{self.__class__.__name__} ({self.department_name})] 正在进行专科分析...")
        knowledge_base = "\n".join(kg_facts) if kg_facts else "本专科无直接相关医学事实记录。"
        
        prompt = f"""
        【患者疑问】: {query}
        【相关医学事实】: 
        {knowledge_base}
        
        请你仅从【{self.department_name}】的专业角度，给出100字左右的专业初步意见。
        如果是复杂疾病可能涉及另外的科室，请提示该情况。
        """
        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": self.specialty_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300,
                name=f"Agent-Dept-{self.department_name}"
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[{self.department_name}] 目前无法提供意见 ({e})。"

# 显式定义的常见6大科室核心类
class InternalMedicineAgent(DepartmentAgent):
    def __init__(self):
        super().__init__("内科")
        self.specialty_prompt += " 请特别关注内科系统性的病变、器官功能异常以及相关的药物治疗方案。"

class SurgeryAgent(DepartmentAgent):
    def __init__(self):
        super().__init__("外科")
        self.specialty_prompt += " 请重点关注是否需要手术干预、外伤处理及术后恢复情况。"

class CardiologyAgent(DepartmentAgent):
    def __init__(self):
        super().__init__("心内科")
        self.specialty_prompt += " 请重点排查心血管系统疾病（如冠心病、高血压、心律失常等）及其紧急性。"

class NeurologyAgent(DepartmentAgent):
    def __init__(self):
        super().__init__("神经内科")
        self.specialty_prompt += " 请重点评估神经系统症状（如头晕、头痛、麻木、昏厥的神经定位体征）。"

class DermatologyAgent(DepartmentAgent):
    def __init__(self):
        super().__init__("皮肤科")
        self.specialty_prompt += " 请重点鉴别皮肤表面皮损形态、传染性及皮肤免疫状态。"

class PediatricsAgent(DepartmentAgent):
    def __init__(self):
        super().__init__("儿科")
        self.specialty_prompt += " 注意患者可能为婴幼儿或儿童，用药剂量和病程发展规律需按儿童生理特点考量。"

class DynamicDepartmentAgent(DepartmentAgent):
    """用于动态生成长尾/不常见科室的专家"""
    def __init__(self, department_name):
        super().__init__(department_name)

class KGAgent:
    """KG检索智能体：从知识图谱获取事实"""
    def __init__(self):
        print(f"[KGAgent] 初始化中，连接地址: {NEO4J_URI}")
        self.parser = QuestionPaser()
        self.searcher = AnswerSearcher()
        
    @observe(as_type="span", name="KGAgent.process")
    def process(self, classified_data):
        print("[KGAgent] 准备查询知识图谱...")
        if not classified_data:
            return None
        
        if not self.searcher.g:
            print("[KGAgent] 错误：知识图谱连接未就绪，无法检索事实。")
            return ["目前无法连接到医疗知识图谱数据库，请检查网络或配置。"]

        try:
            res_sql = self.parser.parser_main(classified_data)
            print(f"  > [KGAgent] 生成 Cypher 查询 (SQL): {res_sql}")
            search_results = self.searcher.search_main(res_sql)
            print(f"  > [KGAgent] 知识图谱查询到 {len(search_results)} 条事实:")
            for idx, fact in enumerate(search_results):
                print(f"    - 事实 {idx+1}: {fact}")
            return search_results
        except Exception as e:
            print(f"[KGAgent] 查询执行出错: {e}")
            return [f"查询知识图谱时发生错误: {e}"]

class ConsultantAgent:
    """建议智能体：LLM 生成回复"""
    def __init__(self):
        self.client = LangfuseOpenAI(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            timeout=60
        )

    @observe(as_type="span", name="ConsultantAgent.process")
    def process(self, query, expert_opinions, history=[]):
        print("[ConsultantAgent] 进行主治会诊与汇总分析...")
        
        # 构造对话上下文
        history_len = len(history)
        print(f"  > [ConsultantAgent] 加载对话历史: {history_len} 条记录 (当前参考最近 5 条)")
        
        system_msg = "你是一名经验丰富的主治医师，你的名字叫 Elipuka。你需要综合多个临床科室专家的会诊意见，为患者提供最终的诊断、建议或进一步的追问。"
        
        if not expert_opinions:
            opinions_text = "目前没有明确的专科意见。"
            print("  > [ConsultantAgent] 未收到专科意见，将基于 LLM 通用医学知识生成回复。")
        else:
            opinions_text = "\n\n".join([f"【{dept}专家意见】:\n{opinion}" for dept, opinion in expert_opinions.items()])
            print(f"  > [ConsultantAgent] 正在整合 {len(expert_opinions)} 个科室的专家意见...")
            
        system_msg += f"\n\n【各科室会诊意见】\n{opinions_text}"

        system_msg += """
        \n【你的任务】
        1. **汇总分析**：综合各科室专家的意见，如果专家意见有交叉或互补，请加以有逻辑的融合。
        2. **追问决策**：如果诸位专家觉得病情依然不明确，请**不要武断确诊**，而是设计一个追问，询问用户是否有能区分核心疾病的关键临床特征或症状。
        3. **生成回复**：
           - 以主治医师亲切、专业的口吻回复患者。
           - 清晰简明地罗列出可能的医学判断或建议进行的医学检查。
           - 严密、通俗，并在末尾附带免责声明："以上建议仅供参考，请以线下医院专业医生的实际诊断为准。"
        """
        
        messages = [{"role": "system", "content": system_msg.strip()}]
        for h in history[-5:]:
            messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": query})
        
        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=0.4,
                max_tokens=2000,
                name="Agent-Consultant"
            )
            # 强制刷新缓冲区，确保记录发送
            from langfuse import Langfuse
            Langfuse().flush()
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM API Error: {e}")
            return "诊断推导中遇到阻碍，请由医生线下确认。"

class MedicalAgentOrchestrator:
    # ... (原有逻辑保持不变)
    def __init__(self):
        self.router = RouterAgent()
        self.triage = TriageAgent()
        self.kg_search = KGAgent()
        self.consultant = ConsultantAgent()
        self.chitchat = ChitChatAgent()

    @observe(name="MedicalAgentOrchestrator.query")
    def query(self, user_input):
        category = self.router.route(user_input)
        if category == "CHITCHAT": return self.chitchat.process(user_input)
        if category == "EMERGENCY": return "情况紧急，请立刻在他人陪同下前往最近医院急诊科，或拨打120急救电话！"
        classified_data = self.triage.process(user_input)
        if not classified_data: return "未能理解您的医疗请求，请换个说法。"
        
        # Legacy/CLI fallback sequence (Not used by LangGraph)
        kg_facts = self.kg_search.process(classified_data)
        
        # Fake department agent execution for legacy CLI
        expert_opinions = {}
        target_depts = classified_data.get("target_departments", ["内科"])
        for dept in target_depts:
            from medical_agents import DynamicDepartmentAgent # dynamic load
            expert_opinions[dept] = DynamicDepartmentAgent(dept).process(user_input, kg_facts)
            
        return self.consultant.process(user_input, expert_opinions)
