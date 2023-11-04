from abc import ABC, abstractmethod

from .message import ChatMessage, AssistantMessage
from .generator import (
    ChatMessageContentGenerator,
    AsyncChatMessageContentGenerator
)

class ChatModel(ABC):
    
    @abstractmethod
    def invoke(self, messages: list[ChatMessage]) -> AssistantMessage:
        """Get the complete response message from the chat model.
        This might take a long time.

        Parameters
        ----------
        messages : list[ChatMessage]
            A list of history messages (including possibly system message).

        Returns
        -------
        AssistantMessage
            Generated response message.
        """
        
        pass
    
    @abstractmethod
    async def ainvoke(self, messages: list[ChatMessage]) -> AssistantMessage:
        """Asynchronously get the complete response message from the chat model.
        This might take a long time.

        Parameters
        ----------
        messages : list[ChatMessage]
            A list of history messages (including possibly system message).

        Returns
        -------
        AssistantMessage
            Generated response message.
        """
        
        pass
    
    @abstractmethod
    def stream(self, messages: list[ChatMessage]) -> ChatMessageContentGenerator:
        """Get the streamed response message from the chat model.

        Parameters
        ----------
        messages : list[ChatMessage]
            A list of history messages (including possibly system message).

        Returns
        -------
        ChatMessageContentGenerator
            A generator that yields a string each time.
        """
        
        pass
    
    @abstractmethod
    async def astream(self, messages: list[ChatMessage]) -> AsyncChatMessageContentGenerator:
        """Asynchronously get the streamed response message from the chat model.

        Parameters
        ----------
        messages : list[ChatMessage]
            A list of history messages (including possibly system message).

        Returns
        -------
        ChatMessageContentGenerator
            An asynchronously generator that yields a string each time.
        """
        
        pass
