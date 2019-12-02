import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from src.model.lgb.validation import learning_curve, val_lists, rank_validation, standard_validation
import cloudpickle

from src.util import get_logger


logger = get_logger(__name__)


def lgb_cv_train_predict(train, train_id, target, target_id, params, model_dir_path, test=None, test_ids=None, n_split=5, thresh=0.5):
    start_time = time.time()    

    assert isinstance(train, pd.DataFrame)
    if test is not None:
        assert isinstance(test, pd.DataFrame)
        test_pred = pd.DataFrame()
        if test_ids is not None:
            test_pred['ID'] = test_ids
            # logger.info('test columns :{}'.format(test.columns))
            # logger.info('test shape :{}'.format(test.shape))

    folds = StratifiedKFold(n_splits=5,shuffle=True)
    cv_models = {}
    logloss, aucs, accuracies, aucs_A, aucs_B, aucs_C, aucs_1, aucs_2, aucs_4, aucs_5, aucs_6, precision, recall, F1_score = val_lists()

    feature_importances = pd.DataFrame()
    feature_importances['feature'] = train.columns
    logger.info('train columns :{}'.format(train.columns))
    # logger.info('train shape :{}'.format(train.shape))
    
    for fold, (train_idx, test_idx) in enumerate(folds.split(train, target)):
        logger.info('Training on fold_{}'.format(fold + 1))

        train_data = lgb.Dataset(train.iloc[train_idx], label=target.iloc[train_idx])
        val_data = lgb.Dataset(train.iloc[test_idx], label=target.iloc[test_idx])

        evals_result={}

        clf = lgb.train(params, train_data, num_boost_round=10000,
                        valid_sets=[train_data, val_data], verbose_eval=100, early_stopping_rounds=50,
                        valid_names=['eval', 'train'],evals_result=evals_result)


        feature_importances['fold_{}'.format(fold + 1)] = clf.feature_importance(importance_type='gain')
        cv_models[fold] = clf
        print(clf.best_score)
        logloss.append(clf.best_score['eval']['binary_logloss'])
        cv_pred = clf.predict(train.iloc[test_idx])
        f_target = target.iloc[test_idx]
        print(pd.DataFrame(cv_pred))
        print(f_target)
        cv_auc, cv_accuracy, cv_precision, cv_recall, cv_f1_score = standard_validation(aucs,accuracies,precision,recall,F1_score, f_target, cv_pred, thresh)
        print('cal start')
        logger.info('Fold_{} val AUC score: {}'.format(fold + 1, cv_auc))
        logger.info('Fold_{} val accuracy: {}'.format(fold + 1, cv_accuracy))
        logger.info('Fold_{} val precision: {}'.format(fold + 1, cv_precision))
        logger.info('Fold_{} val recall: {}'.format(fold + 1, cv_recall))
        logger.info('Fold_{} val f1_score: {}'.format(fold + 1, cv_f1_score))

        #create and output learning curve
        ev_train = pd.DataFrame(evals_result['train']['binary_logloss'])
        ev_test = pd.DataFrame(evals_result['eval']['binary_logloss'])
        learning_curve(ev_train, ev_test, model_dir_path)

        #create_data_for_rank_validation
        train_rank = pd.concat([train_id,train], axis=1)
        target_rank = pd.concat([target_id,target], axis=1)
        train_rank_df = train_rank.iloc[test_idx]
        target_rank_df = target_rank.iloc[test_idx]

        aucs_A, aucs_B, aucs_C, aucs_1, aucs_2, aucs_4, aucs_5, aucs_6 = rank_validation(clf,aucs_A,aucs_B,aucs_C,aucs_1,aucs_2,aucs_4,aucs_5,aucs_6,train_rank_df,target_rank_df)

        #save models
        with open(model_dir_path+'model_{}.pkl'.format(fold + 1), 'wb') as f:
            cloudpickle.dump(clf, f)

        if test is not None:
            test_pred['fold_{}'.format(fold + 1)] = clf.predict(test)

    logger.info('{}-fold training done'.format(n_split))
    logger.info('Mean val logloss: {}'.format(np.mean(logloss)))
    logger.info('Mean val AUC score: {}'.format(np.mean(aucs)))
    logger.info('Mean val RANK A AUC score: {}'.format(np.mean(aucs_A)))
    logger.info('Mean val RANK B AUC score: {}'.format(np.mean(aucs_B)))
    logger.info('Mean val RANK C AUC score: {}'.format(np.mean(aucs_C)))
    logger.info('Mean val man AUC score: {}'.format(np.mean(aucs_1)))
    logger.info('Mean val woman AUC score: {}'.format(np.mean(aucs_2)))
    logger.info('Mean val age 40 AUC score: {}'.format(np.mean(aucs_4)))
    logger.info('Mean val age 50 AUC score: {}'.format(np.mean(aucs_5)))
    logger.info('Mean val age 60 AUC score: {}'.format(np.mean(aucs_6)))
    logger.info('Mean val accuracy score: {}'.format(np.mean(accuracies)))
    logger.info('Mean val precision score: {}'.format(np.mean(precision)))
    logger.info('Mean val recall score: {}'.format(np.mean(recall)))
    logger.info('Mean val F1_score score: {}'.format(np.mean(F1_score)))

    #ToDo！！！
    #va_predを構築する。va_yとindexが同じかどうか注意！
    #create_roc_curve(va_y, va_pred, aucs, model_dir_path)

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

    if test is not None:
        return test_pred