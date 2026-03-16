import os
import sys

# 确保项目根目录在 sys.path 中
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from medical_graph_v2 import medical_graph

def create_initial_state(query=""):
    """
    初始化对话状态字典
    """
    return {
        "query": query,
        "category": "",
        "classified_data": {},
        "kg_facts": [],
        "expert_opinions": {},
        "confirmed_symptoms": [],
        "excluded_diseases": [],
        "history": [],
        "response": ""
    }

def ask_medical_question(question: str, state: dict = None) -> (str, dict):
    """
    统一的医疗模型问答接口。
    
    参数:
    - question: 用户的问题文本
    - state: 当前的对话状态。如果为 None，则自动创建一个新状态。
    
    返回:
    - (回答文本, 更新后的状态字典)
    """
    if state is None:
        state = create_initial_state()
    
    # 更新新问题的输入
    state["query"] = question
    
    # 执行 LangGraph 工作流
    # 注意：LangGraph 的 stream 会逐步返回每个节点的输出，我们取最后的结果
    final_output = None
    try:
        for output in medical_graph.stream(state):
            final_output = output
        
        # 将图中所有节点的输出合并到我们的状态中
        if final_output:
            for node_name, node_output in final_output.items():
                for key, val in node_output.items():
                    state[key] = val
        
        # 确保 response 存在且为字符串
        if "response" not in state or not state["response"]:
            state["response"] = "抱歉，系统暂时无法生成有效回答，请稍后再试。"
            
        # 维护对话历史
        if "history" not in state:
            state["history"] = []
        state["history"].append({"role": "user", "content": question})
        state["history"].append({"role": "assistant", "content": state["response"]})
        
        return state["response"], state
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        error_info = f"核心问答流程执行失败: {str(e)}"
        state["response"] = error_info # 确保状态中也有错误信息
        return error_info, state
