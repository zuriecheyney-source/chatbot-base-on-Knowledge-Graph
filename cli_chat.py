import sys
import os
import io

# 解决 Windows 编码与路径问题
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ask_medical import ask_medical_question, create_initial_state

def cli_chat():
    print("\n" + "="*60)
    print("      医疗知识图谱问诊系统 - 命令行接口 (CLI)")
    print("      (输入 'quit' 退出, 'clear' 重置对话状态)")
    print("="*60 + "\n")
    
    state = create_initial_state()
    
    while True:
        try:
            user_input = input("用户: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("程序已退出。")
                break
                
            if user_input.lower() == 'clear':
                state = create_initial_state()
                print("会话状态已重置。\n")
                continue
            
            # 调用核心问答函数
            print("AI 正在思考 (会诊中)...", end="\r")
            answer, updated_state = ask_medical_question(user_input, state)
            state = updated_state
            
            print(" " * 40, end="\r") # 清除进度提示
            print(f"Elipuka: {answer}\n")
            
        except KeyboardInterrupt:
            print("\n程序终止。")
            break
        except Exception as e:
            print(f"\n[CLI 错误]: {e}")

if __name__ == "__main__":
    cli_chat()
