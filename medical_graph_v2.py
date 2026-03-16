# coding: utf-8
import operator
from typing import Annotated, List, TypedDict, Union

from langgraph.graph import StateGraph, END
from config_advanced import *
from medical_agents import TriageAgent, KGAgent, ConsultantAgent, ChitChatAgent
from langfuse.openai import OpenAI as LangfuseOpenAI
import openai

# 1. 定义问诊流程的状态 (State)
class AgentState(TypedDict):
    query: str                       # 用户原始输入
    category: str                    # 路由类别 (MEDICAL/CHITCHAT/EMERGENCY)
    classified_data: dict             # 意图与实体解析结果
    kg_facts: List[str]              # 检索到的知识片段
    expert_opinions: dict            # 各科室专家的会诊意见
    confirmed_symptoms: Annotated[List[str], operator.add] # 已确认的症状列表 (累积)
    excluded_diseases: Annotated[List[str], operator.add]  # 排除的疾病列表
    history: List[dict]              # 对话历史
    response: str                    # 最终或中间回复

# 2. 初始化智能体组件
from medical_agents import RouterAgent, TriageAgent, KGAgent, ConsultantAgent, ChitChatAgent
from medical_agents import InternalMedicineAgent, SurgeryAgent, CardiologyAgent, NeurologyAgent, DermatologyAgent, PediatricsAgent, DynamicDepartmentAgent

# 缓存已显式实例化的科室 Agent
dept_agents = {
    "内科": InternalMedicineAgent(),
    "外科": SurgeryAgent(),
    "心内科": CardiologyAgent(),
    "神经内科": NeurologyAgent(),
    "皮肤科": DermatologyAgent(),
    "儿科": PediatricsAgent(),
}

router_agent = RouterAgent()
triage_agent = TriageAgent()
kg_agent = KGAgent()
consultant_agent = ConsultantAgent()
chitchat_agent = ChitChatAgent()

# 3. 定义 LangGraph 节点 (Nodes)
def router_node(state: AgentState):
    """路由分发"""
    print("\n" + "="*50)
    print(" [Node: Router] 判断工作流入口...")
    category = router_agent.route(state['query'], state.get('history', []))
    print(f"  > 决策结果: 🚀 {category}")
    return {"category": category}

def triage_node(state: AgentState):
    """提取实体与意图"""
    print(" [Node: Triage] 语义检索与实体解析...")
    res = triage_agent.process(state['query'], state.get('history', []))
    if res is None:
        print("  > [Triage Warning] 意图解析返回为空，将使用空字典兜底。")
        res = {"question_types": [], "args": {}, "target_departments": ["内科"]}
    
    print(f"  > 解析意图: {res.get('question_types', [])}")
    print(f"  > 识别实体: {res.get('args', {})}")
    return {"classified_data": res}

def kg_retrieval_node(state: AgentState):
    """检索知识图谱"""
    print(" [Node: KG Retrieval] 正在访问知识图谱 Neo4j...")
    classified_data = state.get('classified_data', {})
    if not classified_data:
         print("  > [KG Retrieval Warning] 无效的解析数据，跳过检索。")
         return {"kg_facts": []}
         
    facts = kg_agent.process(classified_data)
    print(f"  > [KG Retrieval] 检索环节完成，共注入 {len(facts) if facts else 0} 条事实上下文。")
    return {"kg_facts": facts}

def department_consultation_node(state: AgentState):
    """多科室专家并发会诊"""
    print("\n" + "="*50)
    print(" [Node: Department Consultation] 启动多科室专家会诊...")
    
    target_depts = state.get('classified_data', {}).get('target_departments', ["内科"])
    facts = state.get('kg_facts', [])
    print(f"  > 导诊分配科室: {target_depts}")

    from concurrent.futures import ThreadPoolExecutor
    
    opinions = {}
    
    def run_agent(dept):
        if dept in dept_agents:
            agent = dept_agents[dept]
        else:
            agent = DynamicDepartmentAgent(dept)
            dept_agents[dept] = agent  # 动态缓存长尾科室
        
        opinion = agent.process(state['query'], facts)
        return dept, opinion

    # 使用线程池并发调用 LLM
    with ThreadPoolExecutor(max_workers=len(target_depts)) as executor:
        results = list(executor.map(run_agent, target_depts))
        
    for dept, opinion in results:
        opinions[dept] = opinion
        print(f"  > [{dept}] 会诊意见已生成并收集。")
        
    return {"expert_opinions": opinions}

def diagnosis_node(state: AgentState):
    """主治医师汇总生成回复"""
    print("\n" + "="*50)
    print(" [Node: Consultant Diagnosis] 主治医师汇总诊断逻辑...")
    response = consultant_agent.process(state['query'], state.get('expert_opinions', {}), history=state.get('history', []))
    print(f"  > 最终诊断生成完成，消息长度: {len(response)} 字符")
    print("="*50 + "\n")
    return {"response": response}

def chitchat_node(state: AgentState):
    """闲聊处理"""
    print("\n" + "="*50)
    print(" [Node: ChitChat] 正在进行友好交流...")
    response = chitchat_agent.process(state['query'], state.get('history', []))
    print(f"  > 闲聊回复生成完成")
    print("="*50 + "\n")
    # 添加刷新
    from langfuse import Langfuse
    Langfuse().flush()
    return {"response": response}

# 4. 构建图 (Graph)
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("router", router_node)
workflow.add_node("triage", triage_node)
workflow.add_node("kg_retrieval", kg_retrieval_node)
workflow.add_node("department_consultation", department_consultation_node)
workflow.add_node("diagnosis", diagnosis_node)
workflow.add_node("chitchat", chitchat_node)

# 设置入口
workflow.set_entry_point("router")

# 定义边 (Edges) 与 条件路由
def decide_route(state):
    if "CHITCHAT" in state["category"]:
        return "chitchat"
    elif "EMERGENCY" in state["category"]:
        return "end" # 紧急情况直接结束或走紧急逻辑
    else:
        return "medical"

workflow.add_conditional_edges(
    "router",
    decide_route,
    {
        "chitchat": "chitchat",
        "medical": "triage",
        "end": END
    }
)

workflow.add_edge("triage", "kg_retrieval")
workflow.add_edge("kg_retrieval", "department_consultation")
workflow.add_edge("department_consultation", "diagnosis")
workflow.add_edge("diagnosis", END)
workflow.add_edge("chitchat", END)

# 编译图
medical_graph = workflow.compile()

# 如果需要测试
if __name__ == "__main__":
    inputs = {"query": "感冒了会发烧吗？", "history": []}
    for output in medical_graph.stream(inputs):
        for key, value in output.items():
            print(f"节点 '{key}' 输出完成")
    if 'response' in value:
        print(f"\n最终回答: {value['response']}")
