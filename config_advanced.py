# coding: utf-8
import os

# LangFuse & API Configuration
# 在实际运行前，请确保已安装: pip install langgraph langfuse openai
# 并在此处填入您的 LangFuse 配置（可选）
LANGFUSE_PUBLIC_KEY = "pk-lf-4bcd6918-5c65-41ec-9ab5-029a39a27faa"
LANGFUSE_SECRET_KEY = "sk-lf-790f9424-748f-4815-be6c-a8ae5e176864"
LANGFUSE_HOST = "https://us.cloud.langfuse.com"

# --- 关键：在这里直接设置环境变量，确保全局生效 ---
os.environ["LANGFUSE_PUBLIC_KEY"] = LANGFUSE_PUBLIC_KEY
os.environ["LANGFUSE_SECRET_KEY"] = LANGFUSE_SECRET_KEY
os.environ["LANGFUSE_HOST"] = LANGFUSE_HOST

# 已有的配置同步
from config import *

# 确保导入路径正确
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
