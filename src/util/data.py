import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.util import get_logger


logger = get_logger(__name__)


def load_data(path):
    logger.info('Loading data from {}'.format(path))
    df = pd.read_csv(path)
    logger.info('df shape: {}'.format(df.shape))
    return df

def convert_categorical(train, test=None):
    assert isinstance(train, pd.DataFrame)
    if test is not None:
        assert isinstance(test, pd.DataFrame)

    categoricals = []
    for col in train.columns:
        if train[col].dtype == 'object':
            le = LabelEncoder()
            if test is not None:
                le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))
                test[col] = le.transform(list(test[col].astype(str).values))
            else:
                le.fit(list(train[col].astype(str).values))
            train[col] = le.transform(list(train[col].astype(str).values))
            
            categoricals.append(col)
    logger.info('Categorical variables are converted with: {}'.format(categoricals))
    
    if test is not None:
        return train, test
    else:
        return train
 