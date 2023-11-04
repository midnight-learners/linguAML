from typing import Callable, Any
from abc import ABC
from collections.abc import Generator, AsyncGenerator

class ChatMessageContentGenerator(Generator, ABC):
    
    def __init__(
            self, 
            llm_stream_generator: Generator,
            map_chunk: Callable[[Any], Any],
            stop_criterion: Callable[[Any], bool]
        ) -> None:
        
        self._llm_stream_generator = llm_stream_generator
        self._map_chunk = map_chunk
        self._stop_criterion = stop_criterion
    
    def send(self, value: Any) -> str:
        
        try:
            chunk = next(self._llm_stream_generator)
            content = self._map_chunk(chunk)
            
            if self._stop_criterion(chunk):
                raise StopIteration
            
        except:
            raise StopIteration
        
        return content
        
    def throw(self, typ, val=None, tb=None):
        """Raise an exception in the generator.
        Return next yielded value or raise StopIteration.
        
        Notes
        -----
            This implementation is copied from https://github.com/python/cpython/blob/d5d3249e8a37936d32266fa06ac20017307a1f70/Lib/_collections_abc.py#L309.
        """
        
        if val is None:
            if tb is None:
                raise typ
            val = typ()
            
        if tb is not None:
            val = val.with_traceback(tb)
            
        raise val

class AsyncChatMessageContentGenerator(AsyncGenerator, ABC):
    
    def __init__(
            self, 
            llm_stream_generator: AsyncGenerator,
            map_chunk: Callable[[Any], Any],
            stop_criterion: Callable[[Any], bool]
        ) -> None:
        
        self._llm_stream_generator = llm_stream_generator
        self._map_chunk = map_chunk
        self._stop_criterion = stop_criterion
    
    async def asend(self, value: Any) -> str:
        
        try:
            chunk = await anext(self._llm_stream_generator)
            content = self._map_chunk(chunk)
            
            if self._stop_criterion(chunk):
                raise StopAsyncIteration
            
        except:
            raise StopAsyncIteration
        
        return content
        
    async def athrow(self, typ, val=None, tb=None):
        """Raise an exception in the generator.
        Return next yielded value or raise StopAsyncIteration.
        """
        
        if val is None:
            if tb is None:
                raise typ
            val = typ()
            
        if tb is not None:
            val = val.with_traceback(tb)
            
        raise val
