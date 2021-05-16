#from sklearn import datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import xlrd
import xlwt

read=xlrd.open_workbook('weather.xls')
data=read.sheets()[0]
print(data.nrows)
print(data.ncols)

t1 = data.col_values(11)[1:]    # "Humidity9am"
t1 = np.array( t1).astype(np.float)  # list array  to numpy
len=t1.shape[0]
X=np.reshape(t1, (len,1))
X=np.append(X,np.reshape(np.array(data.col_values(12)[1:]).astype(np.float) ,  (len,1)), axis=1)   # Humidity3pm
X=np.append(X,np.reshape(np.array(data.col_values(4)[1:]).astype(np.float),  (len,1)), axis=1)     # Sunshine
X=np.append(X,np.reshape(np.array(data.col_values(9)[1:]).astype(np.float),  (len,1)), axis=1) #WindSpeed9am     # Sunshine



t1 = data.col_values(20)[1:]    # "Label"
Y = np.array( t1).astype(np.int)  # list array  to numpy
#len=t1.shape[0]
#Y=np.reshape(Y, (1,len))

category=2
dim=X.shape[1]
x_train , x_test , y_train , y_test = train_test_split(X,Y,test_size=0.2)
y_train2=tf.keras.utils.to_categorical(y_train, num_classes=(category))
y_test2=tf.keras.utils.to_categorical(y_test, num_classes=(category))

print("x_train[:4]",x_train[:4])
print("y_train[:4]",y_train[:4])
print("y_train2[:4]",y_train2[:4])

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=100,
    activation=tf.nn.relu,
    input_dim=dim))
model.add(tf.keras.layers.Dense(units=100,
    activation=tf.nn.relu ))
model.add(tf.keras.layers.Dense(units=category,
    activation=tf.nn.softmax ))
model.compile(optimizer='adam',
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=['accuracy'])
model.fit(x_train, y_train2,
          epochs=20000,
          batch_size=64)

model.summary()

score = model.evaluate(x_test, y_test2, batch_size=64)
print("score:",score)

predict2 = model.predict_classes(x_test)
print("predict_classes:",predict2)
print("y_test",y_test[:])


