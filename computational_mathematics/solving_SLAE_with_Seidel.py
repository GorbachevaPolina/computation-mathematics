import numpy as np
import solving_SLAE_with_LU


def generate_matrix_diag(A):
    for i in range(A.shape[0]):
        elem_sum = sum(abs(A[i][j]) for j in range(A.shape[0]))
        A[i][i] = elem_sum
    return A


def generate_matrix_positive(size):
    L = np.array(np.tril(np.random.randint(-100, 100, size=(size, size))), dtype=np.float64)
    return L.dot(L.transpose())



def find_norm_vec(x):
    return max(abs(x[:, 0]))


def find_norm_matrix(matrix):
    norm = -1
    for j in range(len(matrix)):
        curr = sum(abs(matrix[i][j]) for i in range(len(matrix)))
        norm = curr if curr > norm else norm
    return norm


def makeBc(A, b):
    D = np.diag(np.diag(A))
    invD = np.eye(A.shape[0])
    for i in range(A.shape[0]):
        invD[i][i] /= D[i][i]
    B = - invD.dot((A - D))
    c = invD.dot(b)
    return [B, c]


def jacobi(A, b):
    res = makeBc(A, b)
    B = res[0]
    c = res[1]
    x = c
    prev = np.zeros(x.shape)
    eps = 1e-9
    q = find_norm_matrix(B)
    k = 0
    while find_norm_vec(x - prev) > abs(eps * (1-q) / q):
        prev = x
        x = B.dot(prev) + c
        k += 1

    print(k)

    return x


def seidel(A, b):
    n = A.shape[0]
    res = makeBc(A, b)
    B = res[0]
    c = res[1]

    x = np.zeros((n, 1))
    eps = 1e-9
    cont = True
    q = find_norm_matrix(B)
    if q > 1: q = 0.5
    k = 0
    while cont:
        prev = x.copy()
        for i in range(n):
            sum1 = 0
            for j in range(i):
                sum1 += B[i][j] * x[j][0]
            sum2 = 0
            for j in range(i + 1, n):
                sum2 += B[i][j] * prev[j][0]
            x[i][0] = sum1 + sum2 + c[i][0]
        cont = find_norm_vec(x-prev) >= abs(eps * (1-q) / q)
        k += 1

    print(k)
    return x


rows = 3
print('Пусть матрица имеет диагональное преобладание')
A = generate_matrix_diag(np.array(np.random.randint(-100, 100, size=(rows, rows)), dtype=np.float64))
b = np.array(np.random.randint(-100, 100, size=(rows, 1)), dtype=np.float64)
print('A: \n', A, '\nb: \n', b)
res = solving_SLAE_with_LU.LU(A)
L = res[1]
U = res[2]
P = res[3]
x = solving_SLAE_with_LU.solveUx(U, solving_SLAE_with_LU.solveLy(L, P.dot(b)))
print('Метод Якоби: ')
x_jacobi = jacobi(A, b)
print('x: \n', x_jacobi)
print('Проверка: \n', A.dot(x_jacobi)-b)
print(x-x_jacobi)

print('Метод Зейделя: ')
x_seidel = seidel(A, b)
print('x: \n', x_seidel)
print('Проверка: \n', A.dot(x_seidel)-b)
print(x-x_seidel)

print('Пусть матрица положительно определена')
A = generate_matrix_positive(rows)
b = np.array(np.random.randint(-100, 100, size=(rows, 1)), dtype=np.float64)
print('A: \n', A, '\nb: \n', b)

print('Метод Зейделя: ')
x_seidel = seidel(A, b)
print('x: \n', x_seidel)
print('Проверка: \n', A.dot(x_seidel)-b)


