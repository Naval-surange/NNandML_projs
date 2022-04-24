import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow import keras
import tensorflow as tf
import os
os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = keras.utils.normalize(x_train)
x_train = tf.convert_to_tensor(x_train.reshape(-1, 28*28))

x_test = keras.utils.normalize(x_test)
x_test = tf.convert_to_tensor(x_test.reshape(-1, 28*28))

# print(x_train[0])
# plt.imshow(x_train[0])
# plt.show()


model=keras.Sequential(
    [
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(10, activation='softmax')
    ]
)
model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001,),
                loss=keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy']
                )

model.fit(x_train,y_train,batch_size=32,epochs=20,verbose=1)

model.save('digit_classifier.model')


print("Result : ")
model.evaluate(x_test,y_test,batch_size=32,verbose=1)