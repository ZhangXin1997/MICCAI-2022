import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pandas as pd
import logging
from datetime import datetime
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
from scipy import interp
import itertools


def draw_attentionmap(cc_matrix,save_path):
    # cc_matrix = np.abs(array)
    fig, ax = plt.subplots(figsize=(8, 6))
    cc_matrix = cc_matrix.mean(axis=(0, 1))
    # cc_matrix[cc_matrix > 0.2] = 0.2
    im = ax.imshow((cc_matrix + cc_matrix.T) / 2, cmap=plt.cm.RdYlBu)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('', rotation=-90, va="bottom")
    # plt.show()
    plt.savefig(save_path)
    plt.close()

def plt_roc_5fold(preds, targets, save_path):
    fprs, tprs, aucs = [], [], []
    mean_fpr = np.linspace(0, 1, 100)
    for i in range(5):
        fpr, tpr, thresholds = roc_curve(targets[i], preds[i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold%d (AUC = %0.2f)' % (i, roc_auc))
        fprs.append(fpr)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        aucs.append(roc_auc)

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'MeanROC (AUC=%0.4f $\pm$ %0.4f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.show()


def get_logger(outputs_path, name):
    log_level = logging.INFO
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    logging.basicConfig(format='[%(asctime)s-%(levelname)s] %(message)s',
                        level=log_level)
    timenow = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    log_file = os.path.join(outputs_path, 'log_{}_{}.log'.format(name, timenow))

    file_handler = logging.FileHandler(log_file, 'w')
    file_handler.setFormatter(logging.Formatter('[%(asctime)s-%(levelname)s] %(message)s'))
    file_handler.setLevel(log_level)
    logger.addHandler(file_handler)

    return logger


def visualize_training_history(history, save_path, start_loc=1):
    history = history.iloc[start_loc:, :]
    plt.figure(figsize=(7, 4))
    plt.subplot(211)

    train_loss = history['train_loss'].dropna()
    plt.plot(train_loss.index, train_loss, label='train_loss')
    val_loss = history['val_loss'].dropna()
    plt.plot(val_loss.index, val_loss, label='val_loss')
    # test_loss = history['test_loss'].dropna()
    # plt.plot(test_loss.index, test_loss, label='test_loss')
    plt.legend()

    plt.subplot(212)
    train_acc = history['train_acc'].dropna()
    plt.plot(train_acc.index, train_acc, label='train_acc')
    score_acc = history['val_acc'].dropna()
    plt.plot(score_acc.index, score_acc, label='val_acc')
    # test_score_acc = history['test_acc'].dropna()
    # plt.plot(test_score_acc.index, test_score_acc, label='test_acc')
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    # plt.show()


def plot_confusion_matrix(cm, classes=['negative','positive'], normalize=False, title='confusion matrix', cmap=plt.cm.Blues):
    # plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    plt.axis("equal")

    ax = plt.gca()
    left, right = plt.xlim()
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
        plt.text(j, i, num,
                 verticalalignment='center',
                 horizontalalignment="center",
                 color="white" if num > thresh else "black")

    plt.ylabel('true label')
    plt.xlabel('predict label')

    plt.tight_layout()
    # plt.savefig('method_2.png', transparent=True, dpi=800)

    # plt.show()


if __name__ == '__main__':
    # version_name = 'gcn'
    # SAVE_PATH = 'work_result_gcn_no_mask_graphconv/'
    # for i_fold in range(1):
    #     history_file = SAVE_PATH + f"history_{version_name}_fold{i_fold}.csv"
    #     print(f"show {history_file}")
    #     history = pd.read_csv(history_file)
    #     visualize_training_history(history, save_path=os.path.join(SAVE_PATH, 'history.png'))

    plt.figure(figsize=(8, 4))

    nn = pd.read_csv('work_result_nn/history_gcn_fold0.csv')
    score = nn['val_score'].dropna()
    plt.plot(score.index, score, label=f'feature({score.min():.4f})')

    gcn_mask = pd.read_csv('work_result_gcn_mask_graphconv/history_gcn_fold0.csv')
    score = gcn_mask['val_score'].dropna()
    plt.plot(score.index, score, label=f'graphconv_mask({score.min():.4f})')

    gcn_mask = pd.read_csv('work_result_gcn_no_mask_graphconv/history_gcn_fold0.csv')
    score = gcn_mask['val_score'].dropna()
    plt.plot(score.index, score, label=f'graphconv_nomask({score.min():.4f})')

    gcn_nomask = pd.read_csv('work_result_gcn_mask_attention/history_gcn_fold0.csv')
    score = gcn_nomask['val_score'].dropna()
    plt.plot(score.index, score, label=f'attention_mask({score.min():.4f})')

    gcn_nomask = pd.read_csv('work_result_gcn_no_mask_attention/history_gcn_fold0.csv')
    score = gcn_nomask['val_score'].dropna()
    plt.plot(score.index, score, label=f'attention_nomask({score.min():.4f})')

    plt.legend()
    plt.show()
