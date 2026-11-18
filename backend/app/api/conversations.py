from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
import json
import asyncio
import base64
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import desc
from ..core.database import get_db
from ..models.user import User
from ..models.conversation import Conversation
from ..models.message import Message
from ..schemas.chat import (
    ConversationCreate,
    ConversationUpdate,
    ConversationOut,
    ConversationDetail,
    MessageCreate,
    MessageOut,
)
from ..services.auth_service import get_current_user
from ..services.rag_engine import rag_engine

router = APIRouter(prefix="/conversations", tags=["Conversations"])

@router.get("", response_model=List[ConversationOut])
def get_conversations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    convs = (
        db.query(Conversation)
        .filter(Conversation.user_id == current_user.id)
        .order_by(desc(Conversation.pinned), desc(Conversation.updated_at))
        .all()
    )
    result = []
    for c in convs:
        last_msg = (
            db.query(Message)
            .filter(Message.conversation_id == c.id)
            .order_by(desc(Message.created_at))
            .first()
        )
        msg_count = db.query(Message).filter(Message.conversation_id == c.id).count()
        c_dict = {
            "id": c.id,
            "title": c.title,
            "pinned": c.pinned,
            "created_at": c.created_at,
            "updated_at": c.updated_at,
            "message_count": msg_count,
            "last_message": last_msg.content if last_msg else None,
        }
        result.append(ConversationOut(**c_dict))
    return result

@router.post("", response_model=ConversationOut)
def create_conversation(
    conv_in: ConversationCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    new_conv = Conversation(
        user_id=current_user.id,
        title=conv_in.title or "Cuộc trò chuyện mới"
    )
    db.add(new_conv)
    db.commit()
    db.refresh(new_conv)
    return ConversationOut(
        id=new_conv.id,
        title=new_conv.title,
        pinned=new_conv.pinned,
        created_at=new_conv.created_at,
        updated_at=new_conv.updated_at,
        message_count=0,
        last_message=None
    )

@router.get("/{conversation_id}", response_model=ConversationDetail)
def get_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    conv = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id)
        .first()
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Không tìm thấy cuộc trò chuyện này.")
    
    messages = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
        .all()
    )
    
    last_msg = messages[-1].content if messages else None

    return ConversationDetail(
        id=conv.id,
        title=conv.title,
        pinned=conv.pinned,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        message_count=len(messages),
        last_message=last_msg,
        messages=[MessageOut.model_validate(m) for m in messages]
    )

@router.patch("/{conversation_id}", response_model=ConversationOut)
def update_conversation(
    conversation_id: str,
    conv_in: ConversationUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    conv = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id)
        .first()
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Không tìm thấy cuộc trò chuyện này.")

    if conv_in.title is not None:
        conv.title = conv_in.title
    if conv_in.pinned is not None:
        conv.pinned = conv_in.pinned
    conv.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(conv)

    msg_count = db.query(Message).filter(Message.conversation_id == conv.id).count()
    last_msg = (
        db.query(Message)
        .filter(Message.conversation_id == conv.id)
        .order_by(desc(Message.created_at))
        .first()
    )

    return ConversationOut(
        id=conv.id,
        title=conv.title,
        pinned=conv.pinned,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        message_count=msg_count,
        last_message=last_msg.content if last_msg else None
    )

@router.delete("")
def delete_all_conversations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    convs = db.query(Conversation).filter(Conversation.user_id == current_user.id).all()
    for c in convs:
        db.delete(c)
    db.commit()
    return {"status": "success", "message": "Đã xóa toàn bộ lịch sử trò chuyện."}

@router.delete("/{conversation_id}")
def delete_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    conv = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id)
        .first()
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Không tìm thấy cuộc trò chuyện này.")

    db.delete(conv)
    db.commit()
    return {"status": "success", "message": "Đã xóa cuộc trò chuyện thành công."}

@router.post("/{conversation_id}/messages", response_model=MessageOut)
def send_message_in_conversation(
    conversation_id: str,
    msg_in: MessageCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    conv = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id)
        .first()
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Không tìm thấy cuộc trò chuyện này.")

    # 1. Save user query message
    user_msg = Message(
        conversation_id=conv.id,
        sender="user",
        content=msg_in.content
    )
    db.add(user_msg)
    db.commit()

    # If first message in conversation and title was default, auto-generate meaningful title
    if conv.title == "Cuộc trò chuyện mới":
        trimmed_title = msg_in.content[:40] + ("..." if len(msg_in.content) > 40 else "")
        conv.title = trimmed_title

    # 2. Execute RAG query
    rag_result = rag_engine.ask(
        query=msg_in.content,
        top_k=msg_in.top_k,
        gender_filter=msg_in.gender_filter,
        category_filter=msg_in.category_filter
    )

    # 3. Save assistant message with RAG metadata
    asst_msg = Message(
        conversation_id=conv.id,
        sender="assistant",
        content=rag_result["answer"],
        retrieved_products=rag_result["products"],
        grounding_status=rag_result["grounding"]["status"],
        matched_terms=rag_result["grounding"]["matched_terms"],
        filters=rag_result["filters"],
        latency=rag_result["latency"],
        is_outfit=rag_result.get("is_outfit", False)
    )
    db.add(asst_msg)
    
    conv.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(asst_msg)

    return MessageOut.model_validate(asst_msg)

@router.post("/{conversation_id}/messages/stream")
async def send_message_stream(
    conversation_id: str,
    msg_in: MessageCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    conv = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id)
        .first()
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Không tìm thấy cuộc trò chuyện này.")

    user_msg = Message(
        conversation_id=conv.id,
        sender="user",
        content=msg_in.content
    )
    db.add(user_msg)
    db.commit()

    if conv.title == "Cuộc trò chuyện mới":
        conv.title = msg_in.content[:40] + ("..." if len(msg_in.content) > 40 else "")
        db.commit()

    rag_result = rag_engine.ask(
        query=msg_in.content,
        top_k=msg_in.top_k,
        gender_filter=msg_in.gender_filter,
        category_filter=msg_in.category_filter
    )

    full_answer = rag_result["answer"]
    asst_msg = Message(
        conversation_id=conv.id,
        sender="assistant",
        content=full_answer,
        retrieved_products=rag_result["products"],
        grounding_status=rag_result["grounding"]["status"],
        matched_terms=rag_result["grounding"]["matched_terms"],
        filters=rag_result["filters"],
        latency=rag_result["latency"],
        is_outfit=rag_result.get("is_outfit", False)
    )
    db.add(asst_msg)
    conv.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(asst_msg)

    async def event_generator():
        meta_event = {
            "type": "retrieved",
            "message_id": asst_msg.id,
            "products": rag_result["products"],
            "grounding_status": rag_result["grounding"]["status"],
            "matched_terms": rag_result["grounding"]["matched_terms"],
            "latency": rag_result["latency"],
            "filters": rag_result.get("filters", {}),
            "is_outfit": rag_result.get("is_outfit", False),
            "title": conv.title
        }
        yield f"data: {json.dumps(meta_event, ensure_ascii=False)}\n\n"

        words = full_answer.split(" ")
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            yield f"data: {json.dumps({'type': 'token', 'token': chunk}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.015)

        yield f"data: {json.dumps({'type': 'done', 'message_id': asst_msg.id}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.post("/{conversation_id}/messages/image", response_model=MessageOut)
async def send_image_message(
    conversation_id: str,
    file: UploadFile = File(...),
    query: Optional[str] = Form(None),
    top_k: int = Form(5),
    gender_filter: Optional[str] = Form(None),
    category_filter: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    conv = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id)
        .first()
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Không tìm thấy cuộc trò chuyện này.")

    image_bytes = await file.read()
    b64_str = base64.b64encode(image_bytes).decode('utf-8')
    mime = file.content_type or "image/jpeg"
    data_url = f"data:{mime};base64,{b64_str}"

    user_text = query.strip() if query and query.strip() else "Tìm kiếm sản phẩm tương tự bằng hình ảnh"

    user_msg = Message(
        conversation_id=conv.id,
        sender="user",
        content=user_text,
        image_url=data_url
    )
    db.add(user_msg)
    db.commit()

    if conv.title == "Cuộc trò chuyện mới":
        conv.title = f"Ảnh: {user_text[:28]}"
        db.commit()

    rag_result = rag_engine.ask_with_image(
        image_bytes=image_bytes,
        text_query=query,
        top_k=top_k,
        gender_filter=gender_filter,
        category_filter=category_filter
    )

    asst_msg = Message(
        conversation_id=conv.id,
        sender="assistant",
        content=rag_result["answer"],
        retrieved_products=rag_result["products"],
        grounding_status=rag_result["grounding"]["status"],
        matched_terms=rag_result["grounding"]["matched_terms"],
        filters=rag_result["filters"],
        latency=rag_result["latency"]
    )
    db.add(asst_msg)
    conv.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(asst_msg)

    return MessageOut.model_validate(asst_msg)
