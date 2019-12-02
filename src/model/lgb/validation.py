import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve

def val_lists():
    logloss = list()
    aucs = list()
    accuracies = list()
    aucs_A = list()
    aucs_B = list()
    aucs_C = list()
    aucs_1 = list()
    aucs_2 = list()
    aucs_4 = list()
    aucs_5 = list()
    aucs_6 = list()
    precision = list()
    recall = list()
    F1_score = list()
    return logloss, aucs, accuracies, aucs_A, aucs_B, aucs_C, aucs_1, aucs_2, aucs_4, aucs_5, aucs_6, precision, recall, F1_score

def learning_curve(ev_train,ev_test,model_dir_path):
    val_train = pd.DataFrame(ev_train)
    val_test = pd.DataFrame(ev_test)

    # plot
    _, ax1 = plt.subplots(figsize=(5, 4))
    ax1.plot(val_test, label='eval logloss', c='r')
    ax1.plot(val_train, label='train logloss', c='b')
    ax1.set_ylabel('logloss')
    ax1.set_xlabel('rounds')
    ax1.legend()

    plt.grid()
    # plt.show()

    # save
    plt.savefig((os.path.join(model_dir_path, 'learning_plot.png')))

def standard_validation(aucs,accuracies,precision,recall,f_score,true_target, pred, thresh):
    # check_fold_auc
    cv_auc = roc_auc_score(true_target, pred)
    aucs.append(cv_auc)
    # check_fold_accuracy
    cv_accuracy = accuracy_score(true_target, pred > thresh)
    accuracies.append(cv_accuracy)
    # check_fold_precision
    cv_precision = precision_score(true_target, pred > thresh)
    precision.append(cv_precision)
    # check_fold_recall
    cv_recall = recall_score(true_target, pred > thresh)
    recall.append(cv_recall)
    # check_fold_f1score
    cv_f1_score = f1_score(true_target, pred > thresh)
    f_score.append(cv_f1_score)
    return cv_auc,cv_accuracy,cv_precision,cv_recall,cv_f1_score

def rank_validation(clf,aucs_A,aucs_B,aucs_C,aucs_1,aucs_2,aucs_4,aucs_5,aucs_6,train_rank_df,target_rank_df):
    # rank sepalated auc
    train_rank_a = train_rank_df.loc[train_rank_df['RANK'] == 'A'].drop(['ID', 'RANK', 'H25_age_round'], axis=1)
    target_rank_a = target_rank_df.loc[target_rank_df['RANK'] == 'A'].drop(
        ['ID', 'RANK', 'H25_age_round', 'H25_sex'], axis=1)
    cv_pred_A = clf.predict(train_rank_a)
    cv_auc_A = roc_auc_score(target_rank_a, cv_pred_A)
    aucs_A.append(cv_auc_A)
    print('complete A')

    train_rank_b = train_rank_df.loc[train_rank_df['RANK'] == 'B'].drop(['ID', 'RANK', 'H25_age_round'], axis=1)
    target_rank_b = target_rank_df.loc[target_rank_df['RANK'] == 'B'].drop(
        ['ID', 'RANK', 'H25_age_round', 'H25_sex'], axis=1)
    cv_pred_B = clf.predict(train_rank_b)
    cv_auc_B = roc_auc_score(target_rank_b, cv_pred_B)
    aucs_B.append(cv_auc_B)
    print('complete B')

    train_rank_c = train_rank_df.loc[train_rank_df['RANK'] == 'C'].drop(['ID', 'RANK', 'H25_age_round'], axis=1)
    target_rank_c = target_rank_df.loc[target_rank_df['RANK'] == 'C'].drop(
        ['ID', 'RANK', 'H25_age_round', 'H25_sex'], axis=1)
    cv_pred_C = clf.predict(train_rank_c)
    cv_auc_C = roc_auc_score(target_rank_c, cv_pred_C)
    aucs_C.append(cv_auc_C)
    print('complete C')

    # sex auc
    train_rank_1 = train_rank_df.loc[train_rank_df['H25_sex'] == 1].drop(['ID', 'RANK', 'H25_age_round'], axis=1)
    target_rank_1 = target_rank_df.loc[target_rank_df['H25_sex'] == 1].drop(
        ['ID', 'RANK', 'H25_age_round', 'H25_sex'], axis=1)
    cv_pred_1 = clf.predict(train_rank_1)
    cv_auc_1 = roc_auc_score(target_rank_1, cv_pred_1)
    aucs_1.append(cv_auc_1)
    print('complete man')

    train_rank_2 = train_rank_df.loc[train_rank_df['H25_sex'] == 2].drop(['ID', 'RANK', 'H25_age_round'], axis=1)
    target_rank_2 = target_rank_df.loc[target_rank_df['H25_sex'] == 2].drop(
        ['ID', 'RANK', 'H25_age_round', 'H25_sex'], axis=1)
    cv_pred_2 = clf.predict(train_rank_2)
    cv_auc_2 = roc_auc_score(target_rank_2, cv_pred_2)
    aucs_2.append(cv_auc_2)
    print('complete woman')

    # age auc
    train_rank_4 = train_rank_df.loc[train_rank_df['H25_age_round'] == 4].drop(['ID', 'RANK', 'H25_age_round'],
                                                                               axis=1)
    target_rank_4 = target_rank_df.loc[target_rank_df['H25_age_round'] == 4].drop(
        ['ID', 'RANK', 'H25_age_round', 'H25_sex'], axis=1)
    cv_pred_4 = clf.predict(train_rank_4)
    cv_auc_4 = roc_auc_score(target_rank_4, cv_pred_4)
    aucs_4.append(cv_auc_4)
    print('complete age 40')

    train_rank_5 = train_rank_df.loc[train_rank_df['H25_age_round'] == 5].drop(['ID', 'RANK', 'H25_age_round'],
                                                                               axis=1)
    target_rank_5 = target_rank_df.loc[target_rank_df['H25_age_round'] == 5].drop(
        ['ID', 'RANK', 'H25_age_round', 'H25_sex'], axis=1)
    cv_pred_5 = clf.predict(train_rank_5)
    cv_auc_5 = roc_auc_score(target_rank_5, cv_pred_5)
    aucs_5.append(cv_auc_5)
    print('complete age 50')

    train_rank_6 = train_rank_df.loc[train_rank_df['H25_age_round'] == 6].drop(['ID', 'RANK', 'H25_age_round'],
                                                                               axis=1)
    target_rank_6 = target_rank_df.loc[target_rank_df['H25_age_round'] == 6].drop(
        ['ID', 'RANK', 'H25_age_round', 'H25_sex'], axis=1)
    cv_pred_6 = clf.predict(train_rank_6)
    cv_auc_6 = roc_auc_score(target_rank_6, cv_pred_6)
    aucs_6.append(cv_auc_6)
    print('complete age 60')

    return aucs_A,aucs_B,aucs_C,aucs_1,aucs_2,aucs_4,aucs_5,aucs_6

def create_roc_curve(va_y, va_pred, model_dir_path):
    #auc
    auc = roc_auc_score(va_y, va_pred)
    
    # FPR, TPR
    fpr, tpr, thresholds = roc_curve(va_y, va_pred)

    # plot ROC curve
    plt.plot(fpr, tpr, label='ROC curve (area = %.2f)' % auc)
    plt.legend()
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)

    # save
    plt.savefig(model_dir_path+'ROC_plot.png')

def standard_validation_for_outcome(true_target, pred, thresh, model_dir_path):
    # check_fold_auc
    cv_auc = roc_auc_score(true_target, pred)
    # print('auc score : {}'.format(cv_auc))
    # check_fold_accuracy
    cv_accuracy = accuracy_score(true_target, pred > thresh)
    # print('accuracy score : {}'.format(cv_accuracy))
    # check_fold_precision
    cv_precision = precision_score(true_target, pred > thresh)
    # print('precision score : {}'.format(cv_precision))
    # check_fold_recall
    cv_recall = recall_score(true_target, pred > thresh)
    # print('recall score : {}'.format(cv_recall))
    # check_fold_f1score
    cv_f1_score = f1_score(true_target, pred > thresh)
    # print('f1 score : {}'.format(cv_f1_score))

    with open(model_dir_path+"result.txt",'w') as f:
        print('auc score : {}'.format(cv_auc))
        print('accuracy score : {}'.format(cv_accuracy))
        print('precision score : {}'.format(cv_precision))
        print('recall score : {}'.format(cv_recall))
        print('f1 score : {}'.format(cv_f1_score))
