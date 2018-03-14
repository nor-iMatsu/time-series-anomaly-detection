# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def main():
    data = np.loadtxt("./TimeSeriesDataset/qtdbsel102.txt",delimiter="\t")

    train_data = data[1:3000, 2]
    test_data = data[3001:6000, 2]

    w = 50 # width
    m = 2
    k = w/2
    L = k/2 # lag
    Tt = test_data.size
    score = np.zeros(Tt)

    for t in range(w+k, Tt-L+1+1):
        tstart = t-w-k+1
        tend = t-1
        X1 = embed(test_data[tstart:tend], w).T[::-1, :] # trajectory matrix
        X2 = embed(test_data[(tstart+L):(tend+L)], w).T[::-1, :] # test matrix
        #print "{id} X1 : {array}".format(id=t, array=X1.shape)

        U1, s1, V1 = np.linalg.svd(X1, full_matrices=True)
        U1 = U1[:,0:m]
        U2, s2, V2 = np.linalg.svd(X2, full_matrices=True)
        U2 = U2[:,0:m]
        #print "{id} U1 : {array}".format(id=t, array=U1.shape)

        U, s, V = np.linalg.svd(U1.T.dot(U2), full_matrices=True)
        sig1 = s[0]
        #print "{id} S : {array}".format(id=t, array=s)
        score[t] = 1 - np.square(sig1)

    # 変化度をmax1にするデータ整形
    mx = np.max(score)
    score = score / mx

    test_for_plot = data[3001:6000, 2]
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()

    p1, = ax1.plot(score, '-b')
    ax1.set_ylabel('degree of change')
    ax1.set_ylim(0, 1.2)
    ax1.set_xlim(0, 3000)
    p2, = ax2.plot(test_for_plot, '-g')
    ax2.set_ylabel('original')
    ax2.set_ylim(0, 12.0)
    plt.title("Singular Spectrum Transformation")
    ax1.legend([p1, p2], ["degree of change", "original"])
    plt.savefig('/Users/hidenori/Documents/intern2017/execute/results/sst.png')
    plt.show()


def embed(lst, dim):
    emb = np.empty((0,dim), float)
    for i in range(lst.size - dim + 1):
        tmp = np.array(lst[i:i+dim]).reshape((1,-1))
        #print "{id} {array}".format(id=i, array=tmp)
        emb = np.append( emb, tmp, axis=0)
    return emb

if __name__ == '__main__':
    main()
