import numpy as np

def rosenbrock(x1, x2) :
    return (1 - x1) ** 2 + 100 * (x2 - x1 ** 2) ** 2

def x1_der_rosenbrock(x1, x2) :
    return -2 * (1 - x1) - 400 * x1 * (x2 - x1 ** 2)

def x2_der_rosenbrock(x1, x2) :
    return  200 * (x2 - x1 ** 2)

def logistic_regression_2(func, dfunc_x1, dfunc_x2) :
    h = 0.001
    x = np.random.rand(2)

    train_num = 1000

    for i in range(train_num) :
        x1 = x[0]
        x2 = x[1]

        der_x = np.array([dfunc_x1(x1, x2), dfunc_x2(x1, x2)])
        der_x = der_x / np.linalg.norm(der_x)

        x = x - h * der_x

        print("train gen. {0}\nf({1}, {2}) = {3}".format(i + 1, x[0], x[1], func(x[0], x[1])))

logistic_regression_2(rosenbrock, x1_der_rosenbrock, x2_der_rosenbrock)