import json
from datetime import datetime
from fastapi import WebSocket
from sqlmodel import Session, select
from ..models.core import TrialResultChatHistory
from .utils import safe_send_json

async def buffer_message_to_redis(redis_client, result_id: int, role: str, content: str, source: str = "new"):
    key = f"chat_buffer:{result_id}"
    message = {
        "role": role,
        "content": content,
        "timestamp": datetime.utcnow().isoformat(),
        "source": source
    }
    redis_client.rpush(key, json.dumps(message))


async def get_chat_history_from_redis(redis_client, result_id: int, limit: int = 20) -> list[dict]:
    key = f"chat_buffer:{result_id}"
    messages = redis_client.lrange(key, -limit * 2, -1)
    chat_history = []
    for msg in messages:
        try:
            chat_history.append(json.loads(msg))
        except json.JSONDecodeError:
            continue
    return chat_history
 
def stream_chat_history_from_redis(redis_client, result_id: int, limit: int = 20):
    key = f"chat_buffer:{result_id}"
    messages = redis_client.lrange(key, -limit * 2, -1)
    for raw in messages:
        try:
            yield json.loads(raw)
        except json.JSONDecodeError:
            continue


async def load_chat_history_from_db_to_redis(db: Session, redis_client, result_id: int, websocket: WebSocket):
    key = f"chat_buffer:{result_id}"
    if redis_client.exists(key):
        return

    messages = db.exec(
        select(TrialResultChatHistory)
        .where(TrialResultChatHistory.result_id == result_id, TrialResultChatHistory.status == True, TrialResultChatHistory.deleted == False).order_by(TrialResultChatHistory.id)).all()
    for msg in messages:
        await buffer_message_to_redis(redis_client, result_id, msg.role, msg.content, source="db")

    chat_to_send = messages[1:][-5:]
    for msg in chat_to_send:
        await safe_send_json(websocket, {
            "role": msg.role,
            "content": msg.content,
            "timestamp": msg.timestamp.isoformat()
        })

async def flush_messages_from_redis_to_db(redis_client, result_id: int, db: Session, websocket: WebSocket):
    key = f"chat_buffer:{result_id}"
    try:
        if messages := redis_client.lrange(key, 0, -1):
            records = []
            for raw_msg in messages:
                try:
                    parsed = json.loads(raw_msg)
                    if parsed.get("source") == "new":
                        records.append(TrialResultChatHistory(
                            result_id=result_id,
                            role=parsed["role"],
                            content=parsed["content"],
                            timestamp=parsed["timestamp"],
                        ))
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Failed to parse message from Redis: {e}")

            if records:
                try:
                    db.add_all(records)
                    db.commit()
                except Exception as db_error:
                    db.rollback()
                    print(f"DB error during flush: {db_error}")
        redis_client.delete(key)
    except Exception as redis_error:
        print(f"Redis error during flush: {redis_error}")



async def flush_messages_from_redis(redis_client, result_id: int, websocket: WebSocket):
    key = f"chat_buffer:{result_id}"
    try:
        redis_client.delete(key)
        await websocket.send_json({
            "status": "completed",
            "message": "Conversation has ended successfully.",
            "timestamp": datetime.utcnow().isoformat()
        })
        try:
            await websocket.close(code=1000)
        except Exception as close_err:
            print(f"WebSocket may already be closed: {close_err}")
    except Exception as redis_error:
        print(f"Redis error during flush: {redis_error}")