# coding: utf-8
from config_advanced import *
import os

# 1. 必须在导入任何 LangFuse 相关库之前设置环境变量
os.environ["LANGFUSE_PUBLIC_KEY"] = LANGFUSE_PUBLIC_KEY
os.environ["LANGFUSE_SECRET_KEY"] = LANGFUSE_SECRET_KEY
os.environ["LANGFUSE_HOST"] = LANGFUSE_HOST

import openai
from langfuse.openai import OpenAI as LangfuseOpenAI

def test_connection():
    print("--- LangFuse 连接测试 ---")
    print(f"Host: {os.environ.get('LANGFUSE_HOST')}")
    print(f"Public Key: {os.environ.get('LANGFUSE_PUBLIC_KEY')[:10]}...")
    
    client = LangfuseOpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL
    )
    
    print("\n正在尝试发起一次追踪测试对话...")
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": "LangFuse Test Connection"}],
            name="Connection-Test" # 显式命名便于在 Trace 中查找
        )
        print("LLM 响应成功！")
        print(f"响应内容: {response.choices[0].message.content[:30]}...")
        
        # 强制刷向服务器
        print("\n正在强制刷新缓存数据到 LangFuse 服务器...")
        from langfuse import Langfuse
        Langfuse().flush()
        print("刷新指令已发出。请前往 LangFuse 面板查看是否有 'Connection-Test' 记录。")
        
    except Exception as e:
        print(f"测试失败: {e}")

if __name__ == "__main__":
    test_connection()
