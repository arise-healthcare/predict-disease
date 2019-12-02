import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

from src.util import get_logger


logger = get_logger(__name__)


def lgb_cv_train_predict(train, target, params, model_dir_path, test=None, test_ids=None, n_split=5, thresh=0.5):
    start_time = time.time()    

    assert isinstance(train, pd.DataFrame)
    if test is not None:
        assert isinstance(test, pd.DataFrame)
        test_pred = pd.DataFrame()
        if test_ids is not None:
            test_pred['ID'] = test_ids

    folds = StratifiedKFold(n_splits=5)
    cv_models = {}
    logloss = list()
    aucs = list()
    accuracies = list()
    feature_importances = pd.DataFrame()
    feature_importances['feature'] = train.columns
    
    for fold, (train_idx, test_idx) in enumerate(folds.split(train, target)):
        logger.info('Training on fold_{}'.format(fold + 1))
        
        train_data = lgb.Dataset(train.iloc[train_idx], label=target.iloc[train_idx])
        val_data = lgb.Dataset(train.iloc[test_idx], label=target.iloc[test_idx])
        clf = lgb.train(params, train_data, num_boost_round=10000, valid_sets=[train_data, val_data], verbose_eval=100, early_stopping_rounds=50)
        
        feature_importances['fold_{}'.format(fold + 1)] = clf.feature_importance(importance_type='gain')
        cv_models[fold] = clf
        logloss.append(clf.best_score['valid_1']['binary_logloss'])
        cv_pred = clf.predict(train.iloc[test_idx])
        cv_auc = roc_auc_score(target.iloc[test_idx], cv_pred)
        aucs.append(cv_auc)
        cv_accuracy = accuracy_score(target.iloc[test_idx], cv_pred>thresh)
        accuracies.append(cv_accuracy)
        logger.info('Fold_{} val AUC score: {}'.format(fold + 1, cv_auc))
        logger.info('Fold_{} val accuracy: {}'.format(fold + 1, cv_accuracy))

        if test is not None:
            test_pred['fold_{}'.format(fold + 1)] = clf.predict(test)

    logger.info('{}-fold training done'.format(n_split))
    logger.info('Mean val logloss: {}'.format(np.mean(logloss)))
    logger.info('Mean val AUC score: {}'.format(np.mean(aucs)))
    logger.info('Mean val accuracy score: {}'.format(np.mean(accuracies)))

    # Average prediction from k-fold models
    if test is not None:
        test_pred['average'] = test_pred[['fold_{}'.format(fold + 1) for fold in range(folds.n_splits)]].mean(axis=1)
        test_pred_path = os.path.join(model_dir_path, 'test_prediction.csv')
        test_pred.to_csv(test_pred_path, index=False)
        logger.info('Test prediction with shape {} saved'.format(test_pred.shape))
    
    # Feature importance based on split gain
    feature_importances['average'] = feature_importances[['fold_{}'.format(fold + 1) for fold in range(n_split)]].mean(axis=1)
    feature_importance_path = os.path.join(model_dir_path, 'feature_importance_lgb.csv')
    feature_importances.to_csv(feature_importance_path, index=False)
    n_feature_show = np.min([30, train.shape[1]])
    plt.figure(figsize=(12, 12))
    sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(n_feature_show), x='average', y='feature')
    plt.title('{} TOP feature importance over {} folds average'.format(n_feature_show, n_split))
    plt.savefig(os.path.join(model_dir_path, 'feature_importance_lgb.png'))

    # Save models and column names
    for fold in range(n_split):
        model = cv_models[fold]
        model.save_model(os.path.join(model_dir_path, 'model_{}.txt'.format(fold+1)))
    np.save(os.path.join(model_dir_path, 'lgb_columns.npy'), train.columns.values)
    logger.info('Model saved to {}'.format(model_dir_path))

    end_time = time.time()
    t_sec = round(end_time - start_time)
    (t_min, t_sec) = divmod(t_sec,60)
    (t_hour,t_min) = divmod(t_min,60) 
    logger.info('Training done, took {}h:{}m:{}s'.format(t_hour,t_min,t_sec))
