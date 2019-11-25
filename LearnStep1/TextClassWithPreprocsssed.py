from __future__ import absolute_import,division,print_function,unicode_literals
import tensorflow as tf

from tensorflow import keras
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
import numpy as np

print(tf.__version__)

#下载IMDB数据集，已经被tfds预处理
(train_data,test_data),info=tfds.load(
    'imdb_reviews/subwords8k',
    split=(tfds.Split.TRAIN,tfds.Split.TEST),
    as_supervised=True,
    with_info=True
)

#试用解码器
encoder=info.features['text'].encoder
print('Vocabulary size:{}'.format(encoder.vocab_size))

#这个解码器可以反解码任何字符串

sample_string="Hello Tensorflow."
encoded_string=encoder.encode(sample_string)

print ('Encoded string is {}'.format(encoded_string))

#data的格式：这个dataset已经预先处理，每个例子是一个整数数组。
#label是0或1，0表示负面的，1表示正面的

for train_example,train_label in train_data.take(1):
    print('Encode text:',train_example[:10].numpy())
    print('Label:',train_label.numpy())

#创建训练集的批次，由于评论的长度不一致，需要使用padded_batch 补零
BUFFER_SIZE=1000
train_batches=(
    train_data.shuffle(BUFFER_SIZE)
    .padded_batch(32,train_data.output_shapes)
)

test_batches=(
    test_data.
    padded_batch(32,train_data.output_shapes)
)

model = keras.Sequential([
  keras.layers.Embedding(encoder.vocab_size, 16),
  keras.layers.GlobalAveragePooling1D(),
  keras.layers.Dense(1, activation='sigmoid')])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_batches,
                    epochs=10,
                    validation_data=test_batches,
                    validation_steps=30)


loss, accuracy = model.evaluate(test_batches)

print("Loss: ", loss)
print("Accuracy: ", accuracy)


history_dict = history.history
history_dict.keys()

import matplotlib.pyplot as plt

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()