import numpy as np
import math


def QR(A):
    n = A.shape[0]
    r = A.copy()
    q = np.eye(n)
    for j in range(n):
        for i in range(j+1, n):
            tmp = np.eye(n)
            c = r[j][j] / math.sqrt(r[j][j] ** 2 + r[i][j] ** 2)
            s = r[i][j] / math.sqrt(r[j][j] ** 2 + r[i][j] ** 2)
            tmp[j][j] = c
            tmp[j][i] = s
            tmp[i][j] = -s
            tmp[i][i] = c

            r = tmp.dot(r)
            q = tmp.dot(q)
    return [q, r]


def solve(matrix, b):
    x = np.zeros(b.shape)
    for i in range(len(b) - 1, -1, -1):
        curr_sum = 0
        for k in range(i + 1, len(b)):
            curr_sum += matrix[i][k] * x[k][0]
        x[i][0] = (b[i][0] - curr_sum) / matrix[i][i]
    return x


rows = 3
A = np.array(np.random.randint(-100, 100, size=(rows, rows)), dtype=np.float64)
b = np.array(np.random.randint(-100, 100, size=(rows, 1)), dtype=np.float64)
print('A: \n', A)
Q = QR(A)[0].transpose()
R = QR(A)[1]
print('Q: \n', Q, '\nR: \n', R)
print('QR: \n', Q.dot(R))
print('A-QR: \n', A-Q.dot(R))

print('Проверка Q на ортонормированность:')
print('Q^(T) * Q:\n', Q.transpose().dot(Q))
for i in range(rows):
    length = 0
    for j in range(rows):
        length += Q[i][j] ** 2
    print('Длина строки ', i, ' равна ', length)

print('Найдем решение СЛАУ: ')
x = solve(R, Q.transpose().dot(b))
print('x: \n', x)
print('Ax-b: \n', A.dot(x)-b)