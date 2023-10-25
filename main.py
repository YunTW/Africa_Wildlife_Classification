import os
import cv2
import random
import numpy as np
from tensorflow.keras.models import load_model

# 隨機選取圖像測試
imgs = []  # Images
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
            imgs.append(image)

img = random.choice(imgs)

# Load Model
model = load_model("africa_wildlife_classification.h5", compile=False)
prediction = model.predict(img.reshape(1, 128, 128, 3))
prediction = np.argmax(prediction)
print(class_label[prediction])

# Show
cv2.imshow("test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
