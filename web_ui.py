import streamlit as st
import time
from ask_medical import ask_medical_question, create_initial_state

# 配置页面
st.set_page_config(
    page_title="Elipuka 智能医疗问诊系统",
    page_icon="🏥",
    layout="wide"
)

# 初始化 Session State (用于跨刷新保存对话)
if "medical_state" not in st.session_state:
    st.session_state.medical_state = create_initial_state()

# 侧边栏
with st.sidebar:
    st.title("🏥 Elipuka 医疗系统")
    st.info("本系统结合了深度学习(TextCNN/CRF)、知识图谱(Neo4j)与多Agent协同技术。")
    
    if st.button("🔄 重置会话"):
        st.session_state.medical_state = create_initial_state()
        st.rerun()
    
    st.divider()
    st.markdown("### 技术栈说明")
    st.caption("- 意图识别: TextCNN")
    st.caption("- 实体抽取: BiLSTM-CRF")
    st.caption("- 知识库: Neo4j Graph DB")
    st.caption("- 调度框架: LangGraph")
    st.caption("- 可观测性: Langfuse")

# 主界面
st.title("🩺 Elipuka 智能科室专家联合会诊平台")
st.markdown("请输入您的症状描述，系统将自动进行**导诊**并召集相关**科室专家**进行会诊分析。")

# 显示聊天历史
for message in st.session_state.medical_state["history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 用户输入区域
if prompt := st.chat_input("例如：我最近胸口疼，还感觉呼吸不顺畅..."):
    # 1. 展示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 调用核心 Agent 引擎
    with st.chat_message("assistant"):
        # 直接使用简洁的状态展示，避免 React DOM 冲突
        status_placeholder = st.empty()
        status_placeholder.info("🔍 系统正在进行多科室会诊，请稍候...")
        
        try:
            start_time = time.time()
            # 调用函数
            answer, updated_state = ask_medical_question(prompt, st.session_state.medical_state)
            # 更新状态
            st.session_state.medical_state = updated_state
            elapsed_time = time.time() - start_time
            
            # 清除加载状态，展示回答
            status_placeholder.empty()
            
            # 简单的打字机效果，使用单次占位符更新
            res_placeholder = st.empty()
            full_res = ""
            # 加快速度或如果是长文章则直接显示
            if len(answer) > 200:
                res_placeholder.markdown(answer)
            else:
                for word in answer:
                    full_res += word
                    res_placeholder.markdown(full_res + "▌")
                    time.sleep(0.005)
                res_placeholder.markdown(answer)
                
            st.caption(f"⏱️ 本次科室协诊耗时: {elapsed_time:.2f}s")
            
        except Exception as e:
            status_placeholder.error(f"抱歉，系统处理时发生错误: {e}")

# 免责声明
st.divider()
st.caption("⚠️ 声明：本系统为毕业设计演示项目，所提供的医疗建议仅供参考，不作为正式临床诊断依据。如有不适请及时前往医院就诊。")
