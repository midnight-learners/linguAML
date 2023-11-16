from typing import Self, Optional
from enum import Enum

# Imports from this package
from linguaml.types import Number, Model
from linguaml.tolearn.hp import HPConfig, CategoricalHP
from .logistic_regression import logistic_regression_family
from .svc import svc_family
from .random_forest_classifier import random_forest_classifier_family
from .lgbm_classifier import lgbm_classifier_family
from .xgb_classifier import xgb_classifier_family

class Family(Enum):
    
    LOGISTIC_REGRESSION = logistic_regression_family
    SVC = svc_family
    RANDOM_FOREST_CLASSIFIER = random_forest_classifier_family
    LGBM_CLASSIFIER = lgbm_classifier_family
    XGB_CLASSIFIER = xgb_classifier_family
    
    @classmethod
    def from_name(cls, name: str) -> Self:
        """Returns the model family based on the given name.

        Parameters
        ----------
        name : str
            Model family name.

        Returns
        -------
        Self
            Model family.
        """
        
        # Convert to upper case
        name = name.strip().upper()
        
        return Family._member_map_[name]
    
    def name(self) -> str:
        """The name of the model family.
        """
        
        return self.value.model_type.__name__
    
    def hp(self) -> HPConfig:
        """The abstract class of the hyperparameter configuration.
        """
        
        return self.value.hp_config_type

    def n_hps(self) -> int:
        """Number of hyperparameters in this model family.
        """
        
        return self.value.hp_config_type.n_hps()
    
    def n_numeric_hps(self) -> int:
        """Number of numeric hyperparameters.
        """
        
        return self.value.hp_config_type.n_numeric_hps()
    
    def n_categorical_hps(self) -> int:
        """Number of categorical hyperparameters.
        """
        
        return self.value.hp_config_type.n_categorical_hps()
    
    def hp_names(self) -> tuple[str]:
        """Names of all hyperparameters.
        """

        return self.value.hp_config_type.hp_names()
    
    def numeric_hp_names(self) -> tuple[str]:
        """Names of numeric hyperparameters.
        """
        
        return self.value.hp_config_type.numeric_hp_names()
    
    def categorical_hp_names(self) -> tuple[str]:
        """Names of categorical hyperparameters.
        """
        
        return self.value.hp_config_type.categorical_hp_names()
    
    def n_levels_in_category(self, categorical_hp_name: str) -> int:
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
        
        return self.value.hp_config_type.n_levels_in_category(categorical_hp_name)
    
    def hp_type(self, name: str) -> Number | type[CategoricalHP]:
        """Data type of the hyperparameter.
        """
        
        return self.value.hp_config_type.hp_type(name)
        
    def define_model(
            self,
            *,
            hp_config: Optional[HPConfig] = None,
            **kwargs
        ) -> Model:
        """Define a model instance. 
        If hp_config is None then the model is constructed by the given keyword arguments.

        Parameters
        ----------
        hp_config : Optional[HPConfig], optional
            Hyperparameter configuration, by default None.

        Returns
        -------
        Model
            Model instance.
        """
        
        if hp_config is not None:
            model = self._define_model_from_hp_config(hp_config)
            return model
        
        model = self.value.model_type(**kwargs)
        
        return model
    
    def _define_model_from_hp_config(
            self, 
            hp_config: HPConfig
        ) -> Model:
        
        # Create a model instance by unpacking the HP dict
        model = self.value.model_type(**hp_config.model_dump())
        
        return model
