from typing import Optional
import numpy as np
from sklearn.model_selection import train_test_split

def train_valid_test_split(
        X: np.ndarray,
        y: np.ndarray,
        valid_size: float = 0.2,
        test_size: float = 0.2,
        random_state: Optional[int] = None,
        shuffle: bool = True
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Split the given dataset (X, y) into
    training, validation and test datasets.

    Parameters
    ----------
    X : np.ndarray
        features
    y : np.ndarray
        labels
    valid_size : float, optional
        proportion of validation dataset,
        it should be >=0 and < 1,
        by default 0.2
    test_size : float, optional
        proportion of test dataset,
        it should be >=0 and < 1,
        by default 0.2
    random_state : Optional[int], optional
        random state for reproduction, by default None
    shuffle : bool, optional
        whether to shuffle the data before splitting, by default True

    Returns
    -------
    dict[str, tuple[np.ndarray, np.ndarray]]
        {
            "train": (X_train, y_train),
            "valid": (X_valid, y_valid),
            "test": (X_test, y_test),
        }
    """
    
    # Calculate size of training dataset
    valid_test_size = valid_size + test_size
    train_size = 1 - valid_test_size
    assert train_size > 0 and train_size <= 1,\
        "size of training dataset must be >0 and <=1"
        
    # There is only the training dataset
    if valid_test_size == 0:
        X_valid = None
        y_valid = None
        X_test = None
        y_test = None
    
    else:
        # Split the training dataset and the remaining dataset
        X_train, X_valid_test, y_train, y_valid_test = train_test_split(
            X, y,
            train_size=train_size,
            random_state=random_state,
            shuffle=shuffle
        )
        
        assert valid_size >= 0 and valid_size <= 1,\
            "size of validation dataset must be >=0 and <=1"
            
        if valid_size > 0:
            # There is no test dataset
            if test_size == 0:
                X_valid = X_valid_test
                y_valid = y_valid_test
                X_test = None
                y_test = None
                
            # Split the validation dataset and test dataset
            else:
                X_valid, X_test, y_valid, y_test = train_test_split(
                    X_valid_test, y_valid_test,
                    train_size=valid_size / valid_test_size,
                    random_state=random_state,
                    shuffle=shuffle
                )
            
        # There is no validation dataset
        else:
            X_valid = None
            y_valid = None
            X_test = X_valid_test
            y_test = y_valid_test
    
    return {
        "train": (X_train, y_train),
        "valid": (X_valid, y_valid),
        "test": (X_test, y_test)
    }
