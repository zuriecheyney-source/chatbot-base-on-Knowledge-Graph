import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import uuid

# 引入核心调用模块
from ask_medical import ask_medical_question, create_initial_state

app = FastAPI(
    title="Elipuka 医疗知识图谱问诊系统 API", 
    description="提供基于深度学习、知识图谱与多 Agent 会诊的医疗咨询接口。"
)

# 内存会话管理 (生产环境建议使用 Redis)
sessions = {}

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    session_id: str
    history: List[dict]

@app.get("/health")
def health():
    return {"status": "ok", "service": "medical-chatbot-api"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # 1. 会话识别与状态获取
    sid = request.session_id if request.session_id else str(uuid.uuid4())
    
    if sid not in sessions:
        sessions[sid] = create_initial_state()
    
    state = sessions[sid]
    
    # 2. 调用核心逻辑
    try:
        answer, updated_state = ask_medical_question(request.question, state)
        
        # 3. 更新内存中的状态
        sessions[sid] = updated_state
        
        return ChatResponse(
            answer=answer,
            session_id=sid,
            history=updated_state.get("history", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    # 使用命令启动: uvicorn api_server:app --host 127.0.0.1 --port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)
