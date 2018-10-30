from __future__ import print_function

import argparse
import numpy as np
import h5py  # モデルの保存用
import keras
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from tensorflow.python.lib.io import file_io  # Google Cloud Storageのデータ読み書きのため
import sys

batch_size = 10
num_classes = 2
epochs = 20

# 四角形のデータ読み込み関数
def load_rectangles_data(filepath):
    print(filepath)
    with file_io.FileIO(filepath, mode='r') as f:
        data = np.loadtxt(f)
    # 画像データの抽出
    x = data[:,0:784]
    # 正解ラベルの抽出
    y = data[:,784]
    return [x,y]

def train_model(train_file='data/',
                job_dir='./tmp/mnist_mlp', **args):

    # ラベル："Horizontal": 横長，"Vertical": 縦長
    label = ['Horizontal', 'Vertical']

    # 四角形の訓練データの読み込み
    [X_train,y_train] = load_rectangles_data(train_file+'rectangles_train.amat')

    # 四角形のテストデータの読み込み
    [X_test,y_test] = load_rectangles_data(train_file+'rectangles_test.amat')

    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # ラベルをクラス数に対応する配列に変更
    # 例：y_train:[0 1 0 0] -> Y_train:[[1 0],[0 1],[1 0],[1 0]]
    Y_train = keras.utils.to_categorical(y_train,num_classes)
    Y_test = keras.utils.to_categorical(y_test,num_classes)

    # 多層パーセプトロンのネットワーク作成
    # 入力を784次元(28x28)で、最終的な出力をクラス数に設定
    model = Sequential()
    model.add(Dense(512, activation='relu',input_dim=784, init='uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu', init='uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax', input_dim=512, init='uniform'))

    model.summary()

    # 2値分類なのでバイナリを選択，最適化アルゴリズムはRMSpropを選択
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
    history = model.fit(X_train, Y_train,
              epochs=epochs,
              batch_size=batch_size,
              verbose=1,
              validation_data=(X_test, Y_test),
              callbacks=[es])

    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # モデルの保存
    model.save('model.h5')

    # Google Cloud Storageのジョブディレクトリにモデルを保存
    with file_io.FileIO('model.h5', mode='r') as input_f:
        with file_io.FileIO(job_dir + '/model.h5', mode='w+') as output_f:
            output_f.write(input_f.read())

if __name__ == '__main__':
    # オプションをパース
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--train-file',
      help='Cloud Storage bucket or local path to training data')
    parser.add_argument(
      '--job-dir',
      help='Cloud storage bucket to export the model and store temp files')
    args = parser.parse_args()
    arguments = args.__dict__
    train_model(**arguments)
