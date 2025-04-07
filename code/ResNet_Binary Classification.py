import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import random


random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

class Args:
    def __init__(self):
        self.data = '../data/cicids2017_total'

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

if __name__ == '__main__':
    start_time = time.time()

    args = Args()
    X_train, X_test, y_train, y_test, class_names = preprocess()
    num_classes = len(np.unique(y_train))

    target_shape = (64, 64, 1)
    num_features = target_shape[0] * target_shape[1] * target_shape[2]

    if X_train.shape[1] < num_features:
        X_train = np.pad(X_train, ((0, 0), (0, num_features - X_train.shape[1])), mode='constant')
        X_test = np.pad(X_test, ((0, 0), (0, num_features - X_test.shape[1])), mode='constant')
    elif X_train.shape[1] > num_features:
        X_train = X_train[:, :num_features]
        X_test = X_test[:, :num_features]

    X_train = X_train.reshape(-1, 64, 64, 1)
    X_test = X_test.reshape(-1, 64, 64, 1)

    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)

    base_model = ResNet50(weights=None, include_top=False, input_shape=(64, 64, 1))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    #model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.009), metrics=['accuracy'])
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.009), metrics=['accuracy'])

    train_start_time = time.time()
    model.fit(X_train, y_train, epochs=5, batch_size=128, validation_split=0.2)
    train_end_time = time.time()

    loss, accuracy = model.evaluate(X_test, y_test)
    predictions = model.predict(X_test)
    predictions_class = np.argmax(predictions, axis=1)
    true_labels_class = np.argmax(y_test, axis=1)

    print('Precision: {:.4f}'.format(precision_score(true_labels_class, predictions_class, average='macro')))
    print('Recall: {:.4f}'.format(recall_score(true_labels_class, predictions_class, average='macro')))
    print('F1-measure: {:.4f}'.format(f1_score(true_labels_class, predictions_class, average='macro')))
    print('Accuracy: {:.4f}'.format(accuracy))
    print(classification_report(true_labels_class, predictions_class, digits=4, target_names=class_names))


    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['figure.titleweight'] = 'bold'
    y_test_single = np.argmax(y_test, axis=1)
    print(classification_report(y_test_single, predictions_class, digits=4, target_names=["abnormal", "normal"]))
    # print(classification_report(y_test, y_pred, digits=4,target_names=["dos","normal","probe","r2l","u2r"]))
    plt.title("ResNet")
    cm = confusion_matrix(y_test_single, predictions_class)
    # cm=pd.DataFrame(cm,columns=["dos","normal","probe","r2l","u2r"],index=["dos","normal","probe","r2l","u2r"])
    # sns.heatmap(cm,fmt="d",cmap="OrRd",annot=True)

    cm = pd.DataFrame(cm, columns=["abnormal", "normal"], index=["abnormal", "normal"])
    sns.heatmap(cm, fmt="d", cmap="Blues", annot=True)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig('ResNet_Binary Classification.svg', format='svg', dpi=300) 
    plt.show()

    end_time = time.time()
    total_runtime = end_time - start_time
    train_time = train_end_time - train_start_time
    print(f"Total Runtime: {total_runtime:.2f} seconds")
    print(f"Training Time: {train_time:.2f} seconds")