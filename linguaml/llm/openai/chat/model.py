from pydantic import BaseModel, ConfigDict
import openai

# Imports from this package
from linguaml.llm.schemas import (
    ChatModel,
    ChatMessage, 
    AssistantMessage, 
    ChatMessageContentGenerator,
    AsyncChatMessageContentGenerator
)
from ..auth import api_key
from .model_names import OpenAIChatModelName

# Set API key
openai.api_key = api_key

class OpenAIChatModel(ChatModel, BaseModel):
    
    model_name: OpenAIChatModelName = OpenAIChatModelName.GPT_3_5_TURBO_16K
    n: int = 1
    temperature: float = 0.0
    
    model_config = ConfigDict(
        protected_namespaces=(),
        extra="allow"
    )
    
    def invoke(self, messages: list[ChatMessage]) -> AssistantMessage:
         
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=list(map(
                lambda message: message.model_dump(),
                messages
            )),
            temperature=self.temperature,
            
            # Other keyword arguments
            **self.model_extra
        )
        
        # Extract the text
        response_text = response.choices[0].message.content
        
        # Wrap into a message
        message = AssistantMessage(response_text)
        
        return message
    
    async def ainvoke(self, messages: list[ChatMessage]) -> AssistantMessage:
         
        response = await openai.ChatCompletion.acreate(
            model=self.model_name,
            messages=list(map(
                lambda message: message.model_dump(),
                messages
            )),
            temperature=self.temperature,
            
            # Other keyword arguments
            **self.model_extra
        )
        
        # Extract the text
        response_text = response.choices[0].message.content
        
        # Wrap into a message
        message = AssistantMessage(response_text)
        
        return message
    
    def stream(self, messages: list[ChatMessage]) -> ChatMessageContentGenerator:
         
        stream_generator = openai.ChatCompletion.create(
            model=self.model_name,
            messages=list(map(
                lambda message: message.model_dump(),
                messages
            )),
            temperature=self.temperature,
            
            # Enable streaming
            stream=True,
            
            # Other keyword arguments
            **self.model_extra
        )
        
        content_generator = ChatMessageContentGenerator(
            stream_generator, 
            map_chunk=lambda chunk: chunk.choices[0].get("delta").get("content"),
            stop_criterion=lambda chunk: chunk.choices[0].get("delta").get("content") is None
        )
        
        return content_generator
    
    async def astream(self, messages: list[ChatMessage]) -> AsyncChatMessageContentGenerator:
         
        stream_generator = await openai.ChatCompletion.acreate(
            model=self.model_name,
            messages=list(map(
                lambda message: message.model_dump(),
                messages
            )),
            temperature=self.temperature,
            
            # Enable streaming
            stream=True,
            
            # Other keyword arguments
            **self.model_extra
        )
        
        content_generator = AsyncChatMessageContentGenerator(
            stream_generator, 
            map_chunk=lambda chunk: chunk.choices[0].get("delta").get("content"),
            stop_criterion=lambda chunk: chunk.choices[0].get("delta").get("content") is None
        )
        
        return content_generator
