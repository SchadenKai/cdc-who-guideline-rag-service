from fastapi import APIRouter


chat_router = APIRouter(prefix="/chat", tags=["chat"])


@chat_router.post("")
def send_message():
    return {"status_code": 200, "message": "Scaffold response"}
