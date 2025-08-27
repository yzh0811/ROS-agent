from pydantic import BaseModel, Field
from typing import List, Optional
import time
import uuid


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    userInput: str
    stream: Optional[bool] = True
    userId: Optional[str] = None
    conversationId: Optional[str] = None
    excelData: Optional[str] = None  # JSON格式的Excel数据
    excelFilename: Optional[str] = None  # Excel文件名


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: List[ChatCompletionResponseChoice]
    system_fingerprint: Optional[str] = None 