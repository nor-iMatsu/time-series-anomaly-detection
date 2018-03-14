# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

def main():
    data = np.loadtxt("./TimeSeriesDataset/qtdbsel102.txt",delimiter="\t")

    train_data = data[1:3000, 2]
    test_data = data[3001:6000, 2]

    width = 100
    nk = 1

    train = embed(train_data, width)
    test = embed(test_data, width)

    neigh = NearestNeighbors(n_neighbors=nk)
    neigh.fit(train)
    d = neigh.kneighbors(test)[0]

    # 距離をmax1にするデータ整形
    mx = np.max(d)
    d = d / mx

    # プロット
    test_for_plot = data[3001+width:6000, 2]
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()

    p1, = ax1.plot(d, '-b')
    ax1.set_ylabel('distance')
    ax1.set_ylim(0, 1.2)
    p2, = ax2.plot(test_for_plot, '-g')
    ax2.set_ylabel('original')
    ax2.set_ylim(0, 12.0)
    plt.title("Nearest Neighbors")
    ax1.legend([p1, p2], ["distance", "original"])
    plt.savefig('./results/knn.png')
    plt.show()


def embed(lst, dim):
    emb = np.empty((0,dim), float)
    for i in range(lst.size - dim + 1):
        tmp = np.array(lst[i:i+dim])[::-1].reshape((1,-1)) #-1...random
        #print "{id} {array}".format(id=i, array=tmp)
        emb = np.append( emb, tmp, axis=0)
    return emb

if __name__ == '__main__':
    main()
