import random
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, Input, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import random
import time
from collections import Counter
from datetime import datetime
import tensorflow as tf
import keras.callbacks
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras import optimizers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, OneHotEncoder
from keras.utils import plot_model
from torch import optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.callbacks import Callback
import seaborn as sns


random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

class Args:
    def __init__(self):
        self.patience = 5
        self.batch_size = 128
        self.hidden_size = 128
        self.time_steps = 1
        self.epochs = 50
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lr = float(1e-3)
        self.data = '../data/cicids2017_total'
        self.model_path = '../../model/keras_lstm_best_15.keras'
        self.class_weight = None

def preprocess():
    data = args.data
    df = pd.read_csv(data, header=None)
    df.columns = [' Destination Port', ' Flow Duration', ' Total Fwd Packets',
                  ' Total Backward Packets', 'Total Length of Fwd Packets',
                  ' Total Length of Bwd Packets', ' Fwd Packet Length Max',
                  ' Fwd Packet Length Min', ' Fwd Packet Length Mean',
                  ' Fwd Packet Length Std', 'Bwd Packet Length Max',
                  ' Bwd Packet Length Min', ' Bwd Packet Length Mean',
                  ' Bwd Packet Length Std', 'Flow Bytes/s', ' Flow Packets/s',
                  ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min',
                  'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Max',
                  ' Fwd IAT Min', 'Bwd IAT Total', ' Bwd IAT Mean', ' Bwd IAT Std',
                  ' Bwd IAT Max', ' Bwd IAT Min', 'Fwd PSH Flags', ' Bwd PSH Flags',
                  ' Fwd URG Flags', ' Bwd URG Flags', ' Fwd Header Length',
                  ' Bwd Header Length', 'Fwd Packets/s', ' Bwd Packets/s',
                  ' Min Packet Length', ' Max Packet Length', ' Packet Length Mean',
                  ' Packet Length Std', ' Packet Length Variance', 'FIN Flag Count',
                  ' SYN Flag Count', ' RST Flag Count', ' PSH Flag Count',
                  ' ACK Flag Count', ' URG Flag Count', ' CWE Flag Count',
                  ' ECE Flag Count', ' Down/Up Ratio', ' Average Packet Size',
                  ' Avg Fwd Segment Size', ' Avg Bwd Segment Size',
                  ' Fwd Header Length.1', 'Fwd Avg Bytes/Bulk', ' Fwd Avg Packets/Bulk',
                  ' Fwd Avg Bulk Rate', ' Bwd Avg Bytes/Bulk', ' Bwd Avg Packets/Bulk',
                  'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', ' Subflow Fwd Bytes',
                  ' Subflow Bwd Packets', ' Subflow Bwd Bytes', 'Init_Win_bytes_forward',
                  ' Init_Win_bytes_backward', ' act_data_pkt_fwd',
                  ' min_seg_size_forward', 'Active Mean', ' Active Std', ' Active Max',
                  ' Active Min', 'Idle Mean', ' Idle Std', ' Idle Max', ' Idle Min',
                  ' Label']

    df[' Label'] = df[' Label'].apply(lambda r: r if r == 'normal' else 'abnormal')
    df = df.groupby(' Label').apply(lambda x: x.sample(frac=0.1, random_state=42)).reset_index(drop=True)

    print(df[' Label'].value_counts())

    print(df.isin([np.inf, -np.inf]).sum().sum())
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    #categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    label_encoder = LabelEncoder()
    df[' Label'] = label_encoder.fit_transform(df[' Label'])
    class_names = label_encoder.classes_

    X = df.drop(columns=[' Label'])
    y = df[' Label']

    print(y.value_counts())
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    print(y.value_counts())

    X_resampled = pd.DataFrame(X_scaled, columns=X.columns, dtype='float32')
    y_resampled = pd.Series(y)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    print(X_train.shape[0])
    print(X_test.shape[0])

    # return X_train, X_test, y_train, y_test, class_names,class_weight_dict
    return X_train, X_test, y_train, y_test, class_names

class SelfAttention(Layer):
    def __init__(self, units):
        super(SelfAttention, self).__init__()
        self.query_dense = Dense(units)
        self.key_dense = Dense(units)
        self.value_dense = Dense(units)
        self.gamma = self.add_weight(name='gamma', shape=[1], initializer='zeros', trainable=True)

    def call(self, x):
        query = self.query_dense(x)
        key = self.key_dense(x)
        value = self.value_dense(x)

        attention_scores = tf.matmul(query, key, transpose_b=True)
        attention_scores = tf.nn.softmax(attention_scores, axis=-1)
        attention_output = tf.matmul(attention_scores, value)

        return self.gamma * attention_output + x

class SelfAttentionModel(Model):
    def __init__(self, num_classes, units=128):
        super(SelfAttentionModel, self).__init__()
        self.dense1 = Dense(units, activation='relu')
        self.attention = SelfAttention(units)
        self.dense2 = Dense(num_classes, activation='softmax')

    def call(self, x):
        x = self.dense1(x)
        x = self.attention(x)
        x = self.dense2(x)
        return x

class model:
    def __init__(self):
        self.net = None

    def train(self, x_train, y_train):
        start_train_time = time.time()  
        num_classes = len(set(y_train))  
        input_shape = (x_train.shape[1],) 

        self.net = SelfAttentionModel(num_classes=num_classes)
        self.net.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        callbacks = [
            ModelCheckpoint(filepath='self_attention_best_model.keras', monitor='val_loss', save_best_only=True)
        ]

        self.net.fit(x_train, y_train, epochs=50, batch_size=128, validation_split=0.2, callbacks=callbacks)
        end_train_time = time.time() 
        train_time = end_train_time - start_train_time 
        print(train_time)

    def test(self, x_test, y_test):
        yy_pred = self.net.predict(x_test, batch_size=args.batch_size, verbose=0)
        y_pred = np.argmax(yy_pred, axis=1)

        y_test_single = y_test.values if isinstance(y_test, pd.Series) else y_test 

        accuracy = accuracy_score(y_test_single, y_pred)
        precision = precision_score(y_test_single, y_pred, average='macro', zero_division=1)
        recall = recall_score(y_test_single, y_pred, average='macro', zero_division=1)
        f1 = f1_score(y_test_single, y_pred, average='macro', zero_division=1)

        print('Precision:{:.4f}'.format(precision))
        print('Recall: {:.4f}'.format(recall))
        print('F1-measure: {:.4f}'.format(f1))
        print('Accuracy:{:.4f}'.format(accuracy))
        print(classification_report(y_test_single, y_pred, digits=4, target_names=class_names))

        class_accuracy = []
        for i in range(len(class_names)):
            class_mask = (y_test_single == i)
            class_accuracy.append(accuracy_score(y_test_single[class_mask], y_pred[class_mask]))

        plt.rcParams['font.weight'] = 'bold'
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['axes.titleweight'] = 'bold'
        plt.rcParams['figure.titleweight'] = 'bold'

        print(classification_report(y_test_single, y_pred, digits=4, target_names=["abnormal", "normal"]))
        # print(classification_report(y_test, y_pred, digits=4,target_names=["dos","normal","probe","r2l","u2r"]))
        plt.title("SelfAttention")
        cm = confusion_matrix(y_test_single, y_pred)

        cm = pd.DataFrame(cm, columns=["abnormal", "normal"], index=["abnormal", "normal"])
        sns.heatmap(cm, fmt="d", cmap="Blues", annot=True)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        #plt.savefig('Self-Attention_Binary Classification.svg', format='svg', dpi=300)  
        plt.show()

if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)

    args = Args()

    start_total_time = time.time() 
    X_train, X_test, y_train, y_test, class_names = preprocess()

    model = model()
    start_time = time.time()
    model.train(X_train, y_train)
    end_total_time = time.time() 
    total_time = end_total_time - start_total_time  
    print(total_time)
    print(args.model_path)
    model.test(X_test, y_test)


