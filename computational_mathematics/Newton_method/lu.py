import numpy as np


def swap(matrix, pivot_idx, i):
    matrix[i], matrix[pivot_idx] = matrix[pivot_idx].copy(), matrix[i].copy()
    return matrix


def LU(a, count):
    n = a.shape[0]
    p = np.eye(n)
    l = np.zeros(a.shape)
    u = a.copy()
    i = 0

    while i != n:
        value = 0
        pivot_idx = -1
        for j in range(i, n):
            if abs(u[j][i]) > value:
                value = abs(u[j][i])
                pivot_idx = j
        if value != 0:
            if pivot_idx != i:
                p = swap(p, pivot_idx, i)
                l = swap(l, pivot_idx, i)
                u = swap(u, pivot_idx, i)

            for k in range(i, n):
                l[k][i] = u[k][i] / u[i][i]
                count += 1
            for k in range(i + 1, n):
                for j in range(0, n):
                    u[k][j] = u[k][j] - l[k][i] * u[i][j]
                    count += 1
            i += 1

    return [l, u, p, count]


def solveLU(l, u, b, count):
    y = np.zeros(b.shape)
    for k in range(len(b)):
        curr_sum = 0
        for i in range(k):
            curr_sum += l[k][i] * y[i][0]
            count += 1
        y[k][0] = b[k][0] - curr_sum

    x = np.zeros(b.shape)
    for i in range(len(b)-1, -1, -1):
        curr_sum = 0
        for k in range(i+1, len(b)):
            curr_sum += u[i][k] * x[k][0]
            count += 1
        x[i][0] = (y[i][0] - curr_sum) / u[i][i]
        count += 1

    return [x, count]


