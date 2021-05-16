import tensorflow as tf
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

x1=np.random.random((100,2))*-4-1
x2=np.random.random((100,2))*4+1
x_train=np.concatenate((x1, x2))

y1=np.zeros((100,), dtype=int)
y2=np.ones((100,), dtype=int)
y_train=np.concatenate((y1, y2))

plt.plot(x_train[:100,0],x_train[:100,1], 'yo')
plt.plot(x_train[100:,0],x_train[100:,1], 'bo')
# plt.show()


model = tf.keras.models.Sequential([
   tf.keras.layers.Dense(units=4, activation=tf.nn.tanh ,input_dim=2),
   tf.keras.layers.Dense(units=2, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=50,
          batch_size=32)
model.summary()

weights, biases = model.layers[0].get_weights()
print(weights)
print(biases)

x = np.linspace(-5,5,100)
plt.axis([-5, 5, -5, 5])
plt.plot(x, (-weights[0][0]*x-biases[0])/weights[1][0], '-r', label='No.1')
plt.plot(x, (-weights[0][1]*x-biases[0])/weights[1][1], '-g', label='No.2')
plt.plot(x, (-weights[0][2]*x-biases[0])/weights[1][2], '-b', label='No.3')
plt.plot(x, (-weights[0][3]*x-biases[0])/weights[1][3], '-y', label='No.4')
plt.title('Graph of y=(-ax-c)/b')
plt.xlabel('x', color='#1C2833')
plt.ylabel('y', color='#1C2833')
plt.legend(loc='upper left')
plt.grid()
plt.show()

