from __future__ import absolute_import,division,print_function,unicode_literals

#Tensorflow and tf.keras
import tensorflow as tf
from tensorflow import keras


# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

##Import tje Fashion MNIST dataset
#导入fashion mnist 的数据文件，有70000个灰白图片，10个类别
#低分辨率，28*28 pixels

#用60000个训练，10000个验证
fashion_mnist=keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data()

#图片的标签有0-9，对应10个种类，定义一个class_names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images.shape
# (60000,28,28)
len(train_labels)
#60000

#处理数据。在训练前必须处理数据，如果你检查了第一个图片，你会发现他的像素值在0-255之间

# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(True)
# plt.show()

#传入神经网络之前，要把这些值设置到0-1之间
train_images=train_images/255
test_images=test_images/255

#验证一下，显示处理过后的前25张图片
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

#建立模型，模型的基础模块就是layer
#一种创建方法
model=keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(10,activation='softmax')
])


#第一个layer,把2维数据整平维1维，变成28*28=784个像素
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

predictions = model.predict(test_images)

predictions[0]

np.argmax(predictions[0])

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()