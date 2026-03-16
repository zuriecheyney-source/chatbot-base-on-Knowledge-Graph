# coding: utf-8
import os

# Neo4j configuration (Local Desktop)
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"  # 请确保该密码与您本地数据库一致

# LLM configuration (Defaulting to OpenAI-compatible API)
# Please update these with your actual provider (e.g., DeepSeek, ZhipuAI, etc.)
LLM_API_KEY = "sk-4wEczCysTyeB2zwRFN6RC1gt1IOVRD80UgwfMUGjzAlfQIsf"
LLM_BASE_URL = "https://yinli.one/v1"
LLM_MODEL = "gpt-4o-mini"

# Agent configurations
AGENT_ROLES = {
    "triage": "导诊智能体 - 负责提取医疗实体和识别问题类型",
    "kg_agent": "KG检索智能体 - 负责在知识图谱中查询医疗事实",
    "consultant": "建议智能体 - 负责将事实转化为通俗易懂的医生建议"
}
