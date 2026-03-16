# coding: utf-8
import sys
import os

# 确保当前目录在路径中
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from medical_agents import MedicalAgentOrchestrator

def main():
    print("====================================================")
    print("   医疗智能助手 Elipuka-LLM (基于知识图谱 + RAG)")
    print("====================================================")
    print("正在初始化医疗智能系统，请稍候...")
    
    try:
        orchestrator = MedicalAgentOrchestrator()
        print("\n系统已就绪！您可以输入医疗咨询或进行简单日常对话。")
        print("(输入 'quit' 或 'exit' 退出程序)\n")
    except Exception as e:
        print(f"\n[错误] 系统初始化失败: {e}")
        return

    while True:
        try:
            query = input("用户: ").strip()
            
            if not query:
                continue
                
            if query.lower() in ['quit', 'exit', '再见', '退出']:
                print("Elipuka: 祝您身体健康，再见！")
                break
            
            print("Elipuka 正在思考...", end="\r")
            response = orchestrator.run(query)
            print(" " * 30, end="\r") # 清除“正在思考”提示
            print(f"Elipuka: {response}\n")
            
        except KeyboardInterrupt:
            print("\nElipuka: 程序强制退出。")
            break
        except Exception as e:
            print(f"Elipuka: 抱歉，处理过程中发生了一点小状况: {e}\n")

if __name__ == "__main__":
    main()
