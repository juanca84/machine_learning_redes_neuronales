import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
import pickle

def unpickle(file):
  with open(file, 'rb') as fo:
    dict = pickle.load(fo, encoding = 'bytes')
    return dict

def Relu(Z):
    return np.maximum(0,Z)

def dRelu(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

data = unpickle('./data_batch_1')

images = data[b'data']
labels = data[b'labels']

matches0 = [i for i,x in enumerate(labels) if x==0]
matches9 = [i for i,x in enumerate(labels) if x==9]

matches0 = matches0[0:100]
matches9 = matches9[0:100]

d0 = [images[i] for i in tuple(matches0)]
d9 = [images[i] for i in tuple(matches9)]

y0 = np.repeat([1,0], 100)
y9 = np.repeat([0,1], 100)

x = np.array(d0+d9)
x = x - np.mean(x)

y = np.concatenate(([y0],[y9]),axis=0).T

max_epoch = 6

# 1 sample only
#x = np.array([[1],[2],[3],[4],[5]])
#y = np.array([[0.8],[0.2]])

# input units
i_u = 3072

# Hidden units
h_u = 7500

# Output units
o_u = 2

# x as input units
std1 = 1/np.sqrt(i_u/2)
std2 = 1/np.sqrt(h_u/2)
w1 = std1*randn(i_u,h_u)
w2 = std2*randn(h_u,o_u)

# defining losses
losses = np.zeros([max_epoch, 1])

# gradient values
v_w1 = 0
v_w2 = 0

# learning rate
lr = 0.000001

for t in range(max_epoch):
  # x: entrada, w1: primera capa
  x_t = Relu(x.dot(w1))
  # y_t: valor inferido, w2: segunda capa
  y_t = x_t.dot(w2)
  # y_t: valor inferido, y: salida del par de entrenamiento
  loss = np.square(y_t - y).sum()
  losses[t] = loss
  print(t, loss, sep=',')

  grad_y_t = 2.0*(y_t-y)
  grad_w2 = x_t.T.dot(grad_y_t)

  grad_x_t = grad_y_t.dot(w2.T)
  grad_w1 = x.T.dot(grad_x_t*dRelu(x_t))

  v_w1 = grad_w1
  v_w2 = grad_w2

  w1 = w1 - lr*v_w1
  w2 = w2 - lr*v_w2

plt.figure(1)
plt.plot(range(max_epoch),losses,'r*')
plt.show()
