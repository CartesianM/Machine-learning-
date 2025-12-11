import numpy as np
#data layer
X = np.array([[1], [2], [3], [4], [5]])  # hours studied
y_real = np.array([[30], [50], [55], [70], [80]]) #average score

#model
lr = 0.001
def linear_reg(X,m,b):
    return m * X + b


#LOSS FUNCTIONS
def lossF(y_real,y_hat):
    return np.mean((y_real - y_hat)**2)

def partial_m(X, y_hat):
    return 2 * np.mean((y_hat-y_real) * X)

def partial_b(y_hat):
    return 2 * np.mean(y_hat - y_real)


#UPDATE FUNCTIONS
def newM(m, y_hat, X):
    return m-(lr*partial_m(X, y_hat))

def newB(b, y_hat):
    return b-(lr*partial_b(y_hat))
m = 7
b = 1

for i in range(1000):
    predicted_list = linear_reg(X,m,b) #feed input into model
    loss = lossF(y_real,predicted_list) #get the error
    print("the loss at phase "+ str(i)+ "is" + str(loss)+"\n")#
    m = newM(m, predicted_list, X)
    b = newB(b, predicted_list)
print("the new m and b are "+ str(m)+" and "+ str(b)+"\n")

                     
