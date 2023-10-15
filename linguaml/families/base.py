from typing import Optional, Iterable
from abc import ABC
from sklearn.base import ClassifierMixin, RegressorMixin
from ..hp import HPConfig, CategoricalHP

Model = ClassifierMixin | RegressorMixin
ModelType = type[ClassifierMixin | RegressorMixin]

class Family(ABC):
    
    model_type: ModelType
    hp_config_type: type[HPConfig]
    
    @classmethod
    def hp(cls) -> type[HPConfig]:
        """The abstract class of the hyperparameter configuration.
        This is equivalent to the attribute `hp_config_type`.
        """
        
        return cls.hp_config_type
    
    @classmethod
    def n_hps(cls) -> int:
        """Number of hyperparameters in this model family.
        """
        
        return cls.hp_config_type.n_hps()
    
    @classmethod
    def n_numeric_hps(cls) -> int:
        """Number of numeric hyperparameters.
        """
        
        return cls.hp_config_type.n_numeric_hps()
    
    @classmethod
    def n_categorical_hps(cls) -> int:
        """Number of categorical hyperparameters.
        """
        
        return cls.hp_config_type.n_categorical_hps()
    
    @classmethod
    def numeric_hp_names(cls) -> tuple[str]:
        """Names of numeric hyperparameters.
        """
        
        return cls.hp_config_type.numeric_hp_names()
    
    @classmethod
    def categorical_hp_names(cls) -> tuple[str]:
        """Names of categorical hyperparameters.
        """
        
        return cls.hp_config_type.categorical_hp_names()
    
    @classmethod
    def n_levels_in_category(cls, categorical_hp_name: str) -> int:
        """Number of levels in the given category.

        Parameters
        ----------
        category : str
            Categorical hyperparameter name.

        Returns
        -------
        int
            Number of levels.
        """
        
        return cls.hp_config_type.n_levels_in_category(categorical_hp_name)
    
    @classmethod
    def hp_type(cls, name: str) -> float | int | type[CategoricalHP]:
        """Data type of the hyperparameter.
        """
        
        return cls.hp_config_type.hp_type(name)
        
    @classmethod
    def define_model(
            cls,
            *,
            hp_config: Optional[HPConfig] = None,
            **kwargs
        ) -> Model:
        
        if hp_config is not None:
            model = cls._define_model_from_hp_config(hp_config)
            return model
        
        model = cls.model_type(**kwargs)
        return model
    
    @classmethod
    def _define_model_from_hp_config(
            cls, 
            hp_config: HPConfig
        ) -> Model:
        
        # Create a model instance by unpacking the HP dict
        model = cls.model_type(**hp_config.model_dump())
        
        return model
    
def define_family_type(
        name: str,
        model_type: ModelType, 
        hp_config_type: type[HPConfig]
    ) -> type[Family]:
    
    return type(
        name,
        (Family,),
        {
            "model_type": model_type,
            "hp_config_type": hp_config_type
        }
    )
    