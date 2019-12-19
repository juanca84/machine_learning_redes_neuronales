import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
import pickle

def unpickle(file):
  with open(file, 'rb') as fo:
    dict = pickle.load(fo, encoding = 'bytes')
    return dict

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

max_epoch = 200

# input units
i_u = 3072

# Hidden units
h_u = 7500
h_u_1 = 3500
h_u_2 = 7500

# Output units
o_u = 2

# x as input units
std1 = 1/np.sqrt(i_u/2)
std2 = 1/np.sqrt(h_u/2)
std3 = 1/np.sqrt(h_u_1/2)
std4 = 1/np.sqrt(h_u_2/2)

w1 = std1*randn(i_u,h_u)
w2 = std2*randn(h_u,h_u_1)
w3 = std3*randn(h_u_1,h_u_2)
w4 = std4*randn(h_u_2,o_u)

# defining losses
losses = np.zeros([max_epoch, 1])

# gradient values
v_w1 = 0
v_w2 = 0
v_w3 = 0
v_w4 = 0

# learning rate
lr = 0.000001

for t in range(max_epoch):
    #primera capa
    Z2 = x.dot(w1)
    A2 = 1/(1+np.exp(-Z2))

    #segunda capa
    Z3 = A2.dot(w2)
    A3 = 1/(1+np.exp(-Z3))

    # tercera capa
    Z4 = A3.dot(w3)
    A4 = 1/(1+np.exp(-Z4))

    # cuarta capa
    Z5 = A4.dot(w4)
    Y_esperado = 1/(1+np.exp(-Z5))

    # y_t: valor inferido, y: salida del par de entrenamiento
    loss = np.square(Y_esperado - y).sum()
    losses[t] = loss
    print(t, loss, sep=',')

    #back propagation
    grad_Y_esperado = 2.0*(Y_esperado-y)
    grad_w4 = A4.T.dot(grad_Y_esperado)

    grad_a_4 = grad_Y_esperado.dot(w4.T)
    grad_w3 = A3.T.dot(grad_a_4*A4*(1-A4))

    grad_a_3 = grad_a_4.dot(w3.T)
    grad_w2 = A2.T.dot(grad_a_3*A3*(1-A3))

    grad_a_2 = grad_a_3.dot(w2.T)
    grad_w1 = x.T.dot(grad_a_2*A2*(1-A2))

    v_w1 = grad_w1
    v_w2 = grad_w2
    v_w3 = grad_w3
    v_w4 = grad_w4

    w1 = w1 - lr*v_w1
    w2 = w2 - lr*v_w2
    w3 = w3 - lr*v_w3
    w4 = w4 - lr*v_w4

plt.figure(1)
plt.plot(range(max_epoch),losses,'r*')
plt.show()
