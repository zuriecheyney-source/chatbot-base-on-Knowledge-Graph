import sys
import os
import io

# Fix encoding issues on Windows
if sys.platform == "win32":
    # Ensure stdout/stderr use utf-8 to avoid surrogate errors
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        # Fallback for older python
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Ensure TensorFlow legacy behavior and Keras compatibility
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'

# 确保当前目录在路径中
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from medical_graph_v2 import medical_graph

def main():
    print("====================================================")
    print("   医疗进阶版助手 (LangGraph + RAG + 诊断追问)")
    print("====================================================")
    
    # 初始化全局状态
    state = {
        "query": "",
        "category": "",
        "classified_data": {},
        "kg_facts": [],
        "confirmed_symptoms": [],
        "excluded_diseases": [],
        "history": [],
        "response": ""
    }

    print("\n系统已就绪！请输入您的症状进行咨询。")
    print("(输入 'quit' 退出，输入 'clear' 重置问诊状态)\n")

    while True:
        try:
            user_input = input("用户: ").strip()
            
            if not user_input: continue
            if user_input.lower() in ['quit', 'exit', '退出']: break
            if user_input.lower() == 'clear':
                state = {k: ([] if isinstance(v, list) else ("" if isinstance(v, str) else {})) for k, v in state.items()}
                print("系统已重置。\n")
                continue

            # 更新输入
            state["query"] = user_input
            
            # 使用 LangGraph 执行工作流
            print("Elipuka 正在分析...", end="\r")
            
            # 运行图并获取最终状态
            final_output = None
            for output in medical_graph.stream(state):
                final_output = output
            
            # 提取图中最新的 response 和 状态
            # 注意：LangGraph 的 stream 会分步返回每个节点的输出
            # 我们需要累积这些输出到我们的本地 state 中
            for node_name, node_output in final_output.items():
                for key, val in node_output.items():
                    state[key] = val
            
            print(" " * 30, end="\r")
            print(f"Elipuka: {state['response']}\n")
            
            # 记录历史
            state["history"].append({"role": "user", "content": user_input})
            state["history"].append({"role": "assistant", "content": state["response"]})

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"系统运行错误: {e}\n")

if __name__ == "__main__":
    main()
