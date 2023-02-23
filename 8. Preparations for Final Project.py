import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 設定音樂數據集路徑
data_path = "/path/to/music/dataset/"

# 設定流派標籤
genre_labels = {
    "metal": 0,
    "ballad": 1
}

# 讀取數據集並準備訓練和測試集


def load_data():
    # 將數據集分成訓練和測試集
    train_data = []
    test_data = []
    for genre in genre_labels.keys():
        genre_path = os.path.join(data_path, genre)
        files = os.listdir(genre_path)
        for i, file in enumerate(files):
            file_path = os.path.join(genre_path, file)
            data = np.load(file_path)
            if i < 10:
                test_data.append((data, genre_labels[genre]))
            else:
                train_data.append((data, genre_labels[genre]))

    # 將訓練和測試集打亂
    np.random.shuffle(train_data)
    np.random.shuffle(test_data)

    # 將音樂數據轉換為 NumPy 數組
    x_train = np.array([x[0] for x in train_data])
    y_train = np.array([x[1] for x in train_data])
    x_test = np.array([x[0] for x in test_data])
    y_test = np.array([x[1] for x in test_data])

    return (x_train, y_train), (x_test, y_test)

# 構建模型


def build_model(input_shape):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=input_shape),
        layers.Dense(32, activation='relu'),
        layers.Dense(2)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])
    return model


# 載入數據集
(x_train, y_train), (x_test, y_test) = load_data()

# 構建模型
model = build_model(input_shape=x_train[0].shape)

# 訓練模型
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# 評估模型
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
print('\nTest accuracy:', test_acc)

# 進行預測
predictions = model.predict(x_test)
