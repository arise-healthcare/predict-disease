import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb

from src.util import get_logger


logger = get_logger(__name__)


def grid_search_lgb(params, grid_params, train, target):
    # Perform grid search
    logger.info('Performing grid search with: {}'.format(grid_params))
    model = lgb.LGBMClassifier(boosting_type='gbdt', objective='binary')
    grid = GridSearchCV(model, 
                        grid_params,
                        verbose=1,
                        n_jobs=-1,
                        cv=4)
    grid.fit(train, target)
    logger.info('Best parameters: {}'.format(grid.best_params_))
    logger.info('Best score: {}'.format(grid.best_score_))

    # Update parameters based on grid search
    for key in grid_params:
        if len(grid_params[key]) > 1:
            params[key] = grid.best_params_[key]
    logger.info('Parameters updated as: {}'.format(params))
    return params
