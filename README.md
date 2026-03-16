# 基于深度学习与知识图谱的医疗问诊系统 (Elipuka)

本项目是一个功能完备的智能医疗咨询系统，专门为毕业设计研发。它结合了传统的深度学习语义解析、基于 Neo4j 的医疗知识图谱，以及现代的大语言模型 (LLM) 多 Agent 协同技术。

---

##  核心特性

- **深度学习语义解析**：
  - **意图识别**：基于 TextCNN 模型，精准判断用户问诊类别。
  - **命名实体识别 (NER)**：基于 BiLSTM-CRF 模型，从句子中提取疾病、症状、药物等关键实体。
- **医疗知识图谱 (Neo4j)**：
  - 存储了数万条关于疾病、症状、科室、药物及治疗方式的关联数据，提供经过论证的医学事实支持。
- **多 Agent 协同系统 (LangGraph)**：
  - **Router Agent**：分发医疗/非医疗请求。
  - **Triage Agent**：导诊与意图解析。
  - **Department Agents**：模拟内科、外科、心内科等多科室专家的专科分析。
  - **Consultant Agent**：综合多方意见，给出最终的主治汇总。
- **高性能架构**：
  - 各科室专家会诊采用**并行处理** (Parallel Execution)，大幅降低响应延迟。
- **三栖接口支持**：
  - 支持 **CLI 命令行**、**RESTful API** 以及 **Streamlit Web 界面**。
- **全链路追踪**：
  - 集成 **Langfuse**，实时监控 Agent 决策链路与响应质量。

---

##  系统架构图

`mermaid
graph TD
    User((用户)) --> Interface[接口层: CLI/API/Web]
    Interface --> Logic[核心逻辑: ask_medical]
    Logic --> Router{Router Agent}
    
    Router -- 闲聊 --> ChitChat[ChitChat Agent]
    Router -- 医疗 --> Triage[Triage Agent]
    
    Triage --> DL[深度学习模块: TextCNN/BiLSTM]
    Triage --> KG[知识图谱: Neo4j]
    
    Triage --> Consultation[多科室专家会诊]
    Consultation --> Dept1[内科专家Agent]
    Consultation --> Dept2[外科专家Agent]
    Consultation --> Dept3[其他科室Agent]
    
    Dept1 & Dept2 & Dept3 --> Final[Consultant Agent: 主治汇总]
    Final --> Response((最终回复))
    ChitChat --> Response
`

---

##  快速启动

### 1. 环境准备
确保已安装 Python 3.9+，并安装依赖：
`ash
pip install -r requirements.txt
`

### 2. 配置环境变量
在项目目录下创建 .env 文件或直接在系统环境中配置：
- LLM_API_KEY: 您的 LLM 密钥
- LLM_BASE_URL: API 地址
- NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD: 图数据库配置
- LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST: 观测平台配置

### 3. 运行不同接口

####  网页可视化端 (推荐演示用)
`ash
streamlit run web_ui.py
`
访问：http://localhost:8501

####  命令行对话端
`ash
python cli_chat.py
`

####  HTTP API 服务
`ash
python api_server.py
`
接口文档可见：http://127.0.0.1:8000/docs

---

##  技术选型

| 模块 | 技术方案 |
| :--- | :--- |
| **深度学习框架** | TensorFlow 2.16+ (适配 tf.compat.v1) |
| **自然语言处理** | TextCNN / BiLSTM-CRF |
| **知识图谱** | Neo4j Graph Database |
| **Agent 编排** | LangGraph |
| **LLM 追踪** | Langfuse SDK v4 |
| **后端/Web** | FastAPI / Streamlit |

---

##  常见问题修复 (Troubleshooting)

- **Keras 3 兼容性**：针对 LSTMCell 报错，系统已底层重构引入 rnn_cell_impl 确保旧模型加载正常。
- **超时优化**：默认设置请求超时为 60s，并采用 ThreadPoolExecutor 并行化专家调用，响应速度较串行版本提升 60%+。

---

##  免责声明
本系统为毕业设计演示项目，所提供的医疗建议仅供科研与学习参考，不作为正式临床诊断依据。如有身体不适，请务必及时前往正规医疗机构就诊。
