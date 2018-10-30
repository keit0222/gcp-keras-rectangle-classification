import numpy as np
from scipy.misc import toimage
from keras.utils import np_utils
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

# 四角形のデータ読み込み関数
def load_rectangles_data(filename):
    data = np.loadtxt(filename)
    # 画像データの抽出
    x = data[:,0:784]
    # 正解ラベルの抽出
    y = data[:,784]
    return [x,y]

# パラメータ設定
nb_classes = 2
epoch = 20
batch = 10

# ラベル："Horizontal": 横長，"Vertical": 縦長
label = ["Horizontal","Vertical"]

# 四角形の訓練データの読み込み
[X_train,y_train] = load_rectangles_data('rectangles_train.amat')

# 四角形のテストデータの読み込み
[X_test,y_test] = load_rectangles_data('rectangles_test.amat')

# ラベルをクラス数に対応する配列に変更
# 例：y_train:[0 1 0 0] -> Y_train:[[1 0],[0 1],[1 0],[1 0]]
Y_train = np_utils.to_categorical(y_train,nb_classes)
Y_test = np_utils.to_categorical(y_test,nb_classes)

# 多層パーセプトロンのネットワーク作成
# 入力を784次元(28x28)で、最終的な出力をクラス数に設定
model = Sequential()
model.add(Dense(512, input_dim=784, init='uniform'))
model.add(Activation('sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(512, init='uniform'))
model.add(Activation('sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, input_dim=512, init='uniform'))
model.add(Activation('softmax'))

# 2値分類なのでバイナリを選択，最適化アルゴリズムはRMSpropを選択
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(X_train, Y_train,
          nb_epoch=epoch,
          batch_size=batch)

# テストデータを使って，作成したモデルと重みの評価
score = model.evaluate(X_test, Y_test, batch_size=batch)
# 今回は正解率92%
print(score)
# テストデータの一部のラベルを予測
classified_labels = model.predict_classes(X_test[0:10,:].reshape(-1,784))

# 表示する大きさを指定
plt.figure(figsize=(20,10))
for i in range(10):
    plt.subplot(2,5,i+1)
    # 画像データに変換
    img = toimage(X_test[i].reshape(28,28))
    plt.imshow(img, cmap='gray')
    plt.title("Class {}".format(label[classified_labels[i]]),fontsize=20)
    plt.axis('off')
plt.show()

# モデルと重みのパラメータを書き出し
model_json = model.to_json()
open('rectangles_architecture.json', 'w').write(model_json)
model.save_weights('rectangles_weights.h5', overwrite=True)
