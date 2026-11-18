from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from ..core.database import Base

class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    conversation_id = Column(String(36), ForeignKey("conversations.id"), nullable=False, index=True)
    sender = Column(String(20), nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    
    # Multimodal RAG metadata
    image_url = Column(Text, nullable=True)
    is_outfit = Column(JSON, nullable=True, default=False)
    retrieved_products = Column(JSON, nullable=True, default=list)
    grounding_status = Column(String(30), nullable=True, default="none")
    matched_terms = Column(JSON, nullable=True, default=list)
    filters = Column(JSON, nullable=True, default=dict)
    latency = Column(JSON, nullable=True, default=dict)
    
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    conversation = relationship("Conversation", back_populates="messages")
