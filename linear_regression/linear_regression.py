import csv_loader
import numpy as np
import matplotlib.pyplot as plt

def make_matrix_with_x(datas, x_idx) :
    X = np.matrix([[1] + [d['x'][x_idx]] for d in datas])
    Y = np.matrix([d['y'] for d in datas])
    return X, Y.transpose([1, 0])

def make_matrix(datas) :
    X = np.matrix([[1] + d['x'] for d in datas])
    Y = np.matrix([d['y'] for d in datas])
    return X, Y.transpose([1, 0])

if __name__ == "__main__" :
    housing = csv_loader.process_housing()

    print(housing)

    datas = []
    for i in range(len(housing)) :
        datas.append({'x': housing[i][:-1], 'y': housing[i][-1]})

    x_idx = 0
    plt.scatter(x=[d['x'][x_idx] for d in datas], y=[d['y'] for d in datas])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('{0}th factor to y scatter'.format(x_idx))

    plt.show()

    X, Y = make_matrix_with_x(datas, x_idx)

    theta_hat = np.linalg.inv(X.transpose([1,0]) * X) * X.transpose([1, 0]) * Y

    print(theta_hat)

    plt.scatter(x=[d['x'][x_idx] for d in datas], y=[d['y'] for d in datas])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('{0}th factor to y scatter and regression'.format(x_idx))

    data_x = [d['x'][x_idx] for d in datas]
    data_y = (np.matrix([[1] + [d['x'][x_idx]] for d in datas]) * theta_hat).tolist()
    plt.scatter(x=data_x, y=(np.matrix([[1] + [d['x'][x_idx]] for d in datas]) * theta_hat).tolist())

    plt.show()