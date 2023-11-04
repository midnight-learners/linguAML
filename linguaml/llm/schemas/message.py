from pydantic import BaseModel

# Imports from this package
from .role import ChatRole

class ChatMessage(BaseModel):
    
    role: ChatRole
    content: str

class SystemMessage(ChatMessage):
  
    def __init__(self, content: str) -> None:
        
        super().__init__(
            role=ChatRole.SYSTEM, 
            content=content
        )

class UserMessage(ChatMessage):
    
    def __init__(self, content: str) -> None:
        
        super().__init__(
            role=ChatRole.USER, 
            content=content
        )

class AssistantMessage(ChatMessage):
    
    def __init__(self, content: str) -> None:
        
        super().__init__(
            role=ChatRole.ASSISTANT, 
            content=content
        )
