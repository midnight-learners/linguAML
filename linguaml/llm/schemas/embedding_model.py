from abc import ABC, abstractmethod

class EmbeddingModel(ABC):
    
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Embedding dimension.
        """
        
        pass
    
    @abstractmethod
    def embed(self, text: str | list[str]) -> list[float] | list[list[float]]:
        """Embed one or multiple texts to vectors.

        Parameters
        ----------
        text : str | list[str]
            One text or a list of texts.

        Returns
        -------
        list[float] | list[list[float]]
            One embedding vector or a list of corresponding embedding vectors.
        """
        
        pass
