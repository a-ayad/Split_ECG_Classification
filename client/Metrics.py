import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import random
import torch

def load_dataset():
    global config
    config = Config()
    seed_everything(config.seed)
    global df_ptbdb
    df_ptbdb = pd.read_csv(
        '/Medical/Normal/ptbdb_abnormal.csv')
    global df_mitbih
    df_mitbih = pd.read_csv('/Medical/Normal/mitbih_train.csv')
    # print(df_ptbdb)
    df_mitbih_train = pd.read_csv(
        '/Medical/Normal/mitbih_train.csv', header=None)
    df_mitbih_test = pd.read_csv(
        '/Medical/Normal/mitbih_test.csv', header=None)
    df_mitbih = pd.concat([df_mitbih_train, df_mitbih_test], axis=0)
    df_mitbih.rename(columns={187: 'class'}, inplace=True)

    global id_to_label
    id_to_label = {
        0: "Normal",
        1: "Artial Premature",
        2: "Premature ventricular contraction",
        3: "Fusion of ventricular and normal",
        4: "Fusion of paced and normal"
    }
    df_mitbih['label'] = df_mitbih.iloc[:, -1].map(id_to_label)
    # print(df_mitbih.info())

    df_mitbih.to_csv('data.csv', index=False)
    config.csv_path = 'data.csv'

    df_mitbih = pd.read_csv(config.csv_path)
    # print(df_mitbih['label'].value_counts())

    global df_mitbih_new
    config.csv_path = '/Medical/Synthetic/mitbih_with_syntetic.csv'
    df_mitbih_new = pd.read_csv(config.csv_path)


def plotten():
    percentages1 = [count / df_mitbih.shape[0] * 100 for count in df_mitbih['label'].value_counts()]
    percentages2 = [count / df_mitbih_new.shape[0] * 100 for count in df_mitbih_new['label'].value_counts()]

    fig, axs = plt.subplots(1, 2, figsize=(18, 8))

    # origin
    sns.countplot(
        x=df_mitbih['label'],
        ax=axs[0],
        palette="bright",
        order=df_mitbih['label'].value_counts().index
    )
    axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=15);
    axs[0].set_title("Before", fontsize=15)

    for percentage, count, p in zip(
            percentages1,
            df_mitbih['label'].value_counts(sort=True).values,
            axs[0].patches):
        percentage = f'{np.round(percentage, 2)}%'
        x = p.get_x() + p.get_width() / 2 - 0.4
        y = p.get_y() + p.get_height()
        axs[0].annotate(str(percentage) + " / " + str(count), (x, y), fontsize=10, fontweight='bold')

    # with synthetic
    sns.countplot(
        x=df_mitbih_new['label'],
        ax=axs[1],
        palette="bright",
        order=df_mitbih_new['label'].value_counts().index
    )
    axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=15);
    axs[1].set_title("After", fontsize=15)

    for percentage, count, p in zip(
            percentages2,
            df_mitbih_new['label'].value_counts(sort=True).values,
            axs[1].patches):
        percentage = f'{np.round(percentage, 2)}%'
        x = p.get_x() + p.get_width() / 2 - 0.4
        y = p.get_y() + p.get_height()
        axs[1].annotate(str(percentage) + " / " + str(count), (x, y), fontsize=10, fontweight='bold')

    # plt.suptitle("Balanced Sampling between classes", fontsize=20, weight="bold", y=1.01)
    plt.savefig('data_dist.png', facecolor='w', edgecolor='w', format='png',
                transparent=False, bbox_inches='tight', pad_inches=0.1)
    plt.savefig('data_dist.svg', facecolor='w', edgecolor='w', format='svg',
                transparent=False, bbox_inches='tight', pad_inches=0.1)
    plt.show()

def ecg_signals():
    N = 5
    samples = [df_mitbih.loc[df_mitbih['class'] == cls].sample(N) for cls in range(N)]
    titles = [id_to_label[cls] for cls in range(5)]

    with plt.style.context("seaborn-white"):
        fig, axs = plt.subplots(3, 2, figsize=(20, 7))
        for i in range(5):
            ax = axs.flat[i]
            ax.plot(samples[i].values[:, :-2].transpose())
            ax.set_title(titles[i])
            # plt.ylabel("Amplitude")

        plt.tight_layout()
        plt.suptitle("ECG Signals", fontsize=20, y=1.05, weight="bold")
        plt.savefig(f"signals_per_class.svg",
                    format="svg", bbox_inches='tight', pad_inches=0.2)

        plt.savefig(f"signals_per_class.png",
                    format="png", bbox_inches='tight', pad_inches=0.2)
        plt.show()

class Config:
    csv_path = ''
    seed = 2021
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    attn_state_path = '/Medical/Synthetic/attn.pth'
    lstm_state_path = '/Medical/Synthetic/lstm.pth'
    cnn_state_path = '/Medical/Synthetic/cnn.pth'

    attn_logs = '~/split-learning/MESL-main/Medical/Synthetic/attn.csv'
    lstm_logs = '~/split-learning/MESL-main/Medical/Synthetic/lstm.csv'
    cnn_logs = '~/split-learning/MESL-main/Medical/Synthetic/cnn.csv'

    train_csv_path = '/Medical/Synthetic/mitbih_with_syntetic_train.csv'
    test_csv_path = '/Medical/Synthetic/mitbih_with_syntetic_test.csv'

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    https://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        #print('\nset_true: {0}'.format(set_true))
        #print('set_pred: {0}'.format(set_pred))
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        #print('tmp_a: {0}'.format(tmp_a))
        acc_list.append(tmp_a)
    return np.mean(acc_list)


def Accuracy(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        temp += sum(np.logical_and(y_true[i], y_pred[i])) / sum(np.logical_or(y_true[i], y_pred[i]))
    return temp / y_true.shape[0]



def Precision(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        if sum(y_true[i]) == 0:
            continue
        temp += sum(np.logical_and(y_true[i], y_pred[i])) / sum(y_true[i])
    return temp / y_true.shape[0]


def Recall(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        if sum(y_pred[i]) == 0:
            continue
        temp += sum(np.logical_and(y_true[i], y_pred[i])) / sum(y_pred[i])
    return temp / y_true.shape[0]

def F1Measure(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        if (sum(y_true[i]) == 0) and (sum(y_pred[i]) == 0):
            continue
        temp+= (2*sum(np.logical_and(y_true[i], y_pred[i])))/ (sum(y_true[i])+sum(y_pred[i]))
    return temp/ y_true.shape[0]
