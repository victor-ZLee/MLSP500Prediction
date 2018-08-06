# open the saved nn model
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.externals import joblib

net = tf.InteractiveSession()
# Add ops to save and restore all the variables.
saver = tf.train.import_meta_graph('SP500model.meta')
saver.restore(net, tf.train.latest_checkpoint('./'))

graph = tf.get_default_graph()
X = graph.get_tensor_by_name("X:0")
Y = graph.get_tensor_by_name("Y:0")
out = graph.get_tensor_by_name("out:0")
mse = graph.get_tensor_by_name("mse:0")

npzfile=np.load('testData.npz')
X_test=npzfile['X_test'];y_test=npzfile['y_test'];data_test=npzfile['data_test']
scaler=joblib.load('scalerData')
#X_test=np.random.random((2,500))
pred = net.run(out, feed_dict={X: X_test})

fig = plt.figure()
ax1 = fig.add_subplot(111)
line1, = ax1.plot(y_test)
line2, = ax1.plot(np.transpose(pred))
plt.legend(('Real','Predicted'))
plt.show()

plt.figure()
tmp=np.zeros(data_test.shape)
tmp[:,0]=pred
origData_test=scaler.inverse_transform(tmp)
predY=origData_test[:,0]
tmp=scaler.inverse_transform(data_test)
plt.plot(tmp[:,0]);plt.plot(predY)
plt.legend(('Real','Predicted'))
