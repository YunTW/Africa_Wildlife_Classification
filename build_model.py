import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.data import Dataset
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
import matplotlib.pyplot as plt

# Step 1: Load data
X = []  # Images
y = []  # Index of class_label
class_label = []

DATASET_DIRECTORY = "./African Wildlife"
for folder in os.listdir(DATASET_DIRECTORY):
    folderPath = os.path.join(DATASET_DIRECTORY, folder)
    if not os.path.isdir(folderPath):
        continue
    class_label.append(folder)
    for fileName in os.listdir(folderPath):
        if fileName.endswith("jpg"):
            filePath = os.path.join(folderPath, fileName)
            image = cv2.imread(filePath)
            image = cv2.resize(image, (128, 128))
            X.append(image)
            y.append(len(class_label) - 1)

# Step 2: Data Preparation
X = np.array(X).astype(np.float32) / 255  # 0.0~1.0
y = np.array(y)

# 分割資料
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=42
)

# 訓練資料集
Training_Data = (
    Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(buffer_size=X_train.shape[0])
    .batch(32)
)


# Step 3: Build Model
def vgg16_block(input_layer):
    # 第一個卷積層
    conv01 = Conv2D(64, (3, 3), activation="relu", padding="same")(input_layer)
    conv02 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv01)
    # max_pooling01 = MaxPooling2D((2, 2), strides=(2, 2))(conv02)
    return MaxPooling2D((2, 2), strides=(2, 2))(conv02)

    # # 第二個卷積層
    # conv03 = Conv2D(128, (3, 3), activation="relu", padding="same")(max_pooling01)
    # conv04 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv03)
    # max_pooling02 = MaxPooling2D((2, 2), strides=(2, 2))(conv04)

    # # 第三個卷積層
    # conv05 = Conv2D(256, (3, 3), activation="relu", padding="same")(max_pooling02)
    # conv06 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv05)
    # conv07 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv06)
    # max_pooling03 = MaxPooling2D((2, 2), strides=(2, 2))(conv07)

    # # 第四個卷積層
    # conv08 = Conv2D(512, (3, 3), activation="relu", padding="same")(max_pooling03)
    # conv09 = Conv2D(512, (3, 3), activation="relu", padding="same")(conv08)
    # conv10 = Conv2D(512, (3, 3), activation="relu", padding="same")(conv09)
    # max_pooling04 = MaxPooling2D((2, 2), strides=(2, 2))(conv10)

    # # 第五個卷積層
    # conv11 = Conv2D(512, (3, 3), activation="relu", padding="same")(max_pooling04)
    # conv12 = Conv2D(512, (3, 3), activation="relu", padding="same")(conv11)
    # conv13 = Conv2D(512, (3, 3), activation="relu", padding="same")(conv12)
    # return MaxPooling2D((2, 2), strides=(2, 2))(conv13)


# 輸入層
input_layer = Input(shape=(128, 128, 3))
# 卷積層
vgg_block = vgg16_block(input_layer)
# 全連接層
flatten_layer = Flatten()(vgg_block)
# dense01 = Dense(4096, activation="relu")(flatten_layer)
# dense02 = Dense(4096, activation="relu")(dense01)
dense01 = Dense(64, activation="relu")(flatten_layer)
dense02 = Dense(64, activation="relu")(dense01)
output_layer = Dense(4, activation="softmax")(dense02)
model = Model(inputs=input_layer, outputs=output_layer)
print(model.summary())

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])

epochs = 10
model_history = model.fit(
    Training_Data, epochs=epochs, validation_data=(X_test, y_test), verbose=1
)

# Save the model as an H5 file
model.save("africa_wildlife_classification.h5")

# Step 4: Evaluate Model
# 空的圖表(6*4)
plt.figure(figsize=(6, 4))
# 繪製訓練損失函數折線圖
plt.plot(
    range(epochs),
    model_history.history["loss"],
    c="blue",
    marker="o",
    label="Training loss",
)
# 繪製驗證損失函數折線圖
plt.plot(
    range(epochs),
    model_history.history["val_loss"],
    c="red",
    marker="x",
    label="Validation loss",
)
# 添加圖例
plt.legend()
# 顯示圖形
# plt.show()
# 儲存圖表
plt.savefig("loss_history.png")

# 空的圖表(6*4)
plt.figure(figsize=(6, 4))
# 繪製訓練損失函數折線圖
plt.plot(
    range(epochs),
    model_history.history["acc"],
    c="blue",
    marker="o",
    label="Training acc",
)
# 繪製驗證損失函數折線圖
plt.plot(
    range(epochs),
    model_history.history["val_acc"],
    c="red",
    marker="x",
    label="Validation acc",
)
# 添加圖例
plt.legend()
# 顯示圖形
# plt.show()
# 儲存圖表
plt.savefig("acc_history.png")
