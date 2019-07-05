import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv

def main():
    df = pd.read_csv("./data.csv")
    datas = df.iloc[:,:].values
    hotelling(datas[:,:])


def hotelling(X):
    X = X.T
    p = X.shape[0]
    n = X.shape[1]

    X_mean = np.mean(X, axis=1)
    X_mean = np.reshape(X_mean,(p,1))


    sx = np.zeros((p,p))
    for i in range(n):
        row_vector = X[:,i:i+1]-X_mean
        sx = sx + np.dot(row_vector, row_vector.T)
    sx = sx/(n-1)

    print(X_mean.shape)

    T_value = []
    for i in range(n):
        row_v = X[:,i:i+1]- X_mean
        value = np.dot(np.dot(row_v.T,inv(sx)), row_v)
        T_value.append(value[0,0])
    
    # alpha = 0.05
    # F = 1.8307
    # alpha = 0.01
    F = 2.321
    F=2.958829845
    F = ( p*(n+1)*(n-1) / (n**2-n*p) )*F
    print(F)

    count = 0
    for i in range(n):
        if T_value[i] > F:
            count = count + 1
    
    print(count,"items can be deleted.")
    plt.plot(T_value)
    plt.plot(list(range(n)),[F]*n)
    plt.show()

if "__main__" == __name__:
    main()