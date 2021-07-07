import numpy as np


# Находим норму заданной матрицы
def find_norm(matrix):
    norm = -1
    for j in range(len(matrix)):
        curr = 0
        for i in range(len(matrix)):
            curr += abs(matrix[i][j])
        norm = curr if curr > norm else norm
    return norm


# находим определитель заданной матрицы через LU разложение
def find_determinant(matrix, swap_count):
    result = (-1) ** swap_count
    for i in range(len(matrix)):
        result *= matrix[i][i]
    return result


# обмен строк местами
def swap(matrix, pivot_idx, i):
    matrix[i], matrix[pivot_idx] = matrix[pivot_idx].copy(), matrix[i].copy()
    return matrix


def swap_cols(matrix, pivot_idx, curr):
    tmp = matrix[:, pivot_idx].copy()
    for i in range(pivot_idx, curr):
        matrix[:, i] = matrix[:, i + 1]
    matrix[:, curr] = tmp
    return matrix


# LU разложение
def LU(a):
    n = a.shape[0]
    p = np.eye(n)
    q = np.eye(n)
    l = np.zeros(a.shape)
    u = a.copy()
    swap_count = 0
    i = 0
    curr = n-1  # номер столбца, на который будем переставлять нулевой стобец
    is_deg = False
    epsilon = find_norm(a) * 100 * np.finfo(float).eps

    while i != n:
        value = 0  # значение максимального элемента в оставшемся столбце
        pivot_idx = -1  # индекс максимального элемента в оставшемся столбце
        for j in range(i, n):
            if abs(u[j][i]) > value:
                value = abs(u[j][i])
                pivot_idx = j
        if value >= epsilon:
            if pivot_idx != i:  # при необходимости меняем строки местами
                p = swap(p, pivot_idx, i)
                l = swap(l, pivot_idx, i)
                u = swap(u, pivot_idx, i)
                swap_count += 1

            for k in range(i, n):
                l[k][i] = u[k][i] / u[i][i]
            for k in range(i+1, n):
                for j in range(0, n):
                    u[k][j] = u[k][j] - l[k][i] * u[i][j]
            i += 1
        else:
            is_deg = True  # Матрица вырождена
            if i < curr:  # если остались ненулевые столбцы, то переставляем столбец
                q = swap_cols(q, i, curr)
                u = swap_cols(u, i, curr)
                curr -= 1
            else:
                # если ненулевых столбцов не осталось,
                # то заполняем оставшуюся диагональ L единицами и заканчиваем разложение
                for k in range(i, n):
                    l[k][k] = 1
                rank = i
                return [is_deg, l, u, p, q, rank]

    return [is_deg, l, u, p, swap_count]


def solveLy(l, b):
    y = np.array(np.zeros(b.shape), dtype=np.float64)
    for k in range(len(b)):
        curr_sum = 0
        for i in range(k):
            curr_sum += l[k][i] * y[i][0]
        y[k][0] = b[k][0] - curr_sum
    return y


def solveUx(u, y):
    x = np.array(np.zeros(y.shape), dtype=np.float64)
    for i in range(len(y) - 1, -1, -1):
        curr_sum = 0
        for k in range(i + 1, len(y)):
            curr_sum += u[i][k] * x[k][0]
        if u[i][i] != 0:
            x[i][0] = (y[i][0] - curr_sum) / u[i][i]
    return x


rows = 3
A = np.array(np.random.randint(-5, 5, size=(rows, rows)), dtype=np.float64)
b = np.array(np.random.randint(-100, 100, size=(rows, 1)), dtype=np.float64)

# for i in range(2, rows-1):
#     A[:, i] = (i-1) * A[:, 0] + (2-i) * A[:, 1]
# b = A.dot(b)

epsilon = find_norm(A) * 100 * np.finfo(float).eps
print('A: \n', A, '\nb: \n', b)
result = LU(A)
is_deg = result[0]
L = result[1]
U = result[2]
P = result[3]
if is_deg:
    Q = result[4]
    rank = result[5]
    print('\nL: \n', L, '\nU: \n', U,'\nP: \n', P, '\nQ: \n', Q)
    print('LU: ', '\n', L.dot(U))
    print('PAQ: ', '\n', P.dot(A.dot(Q)))
    print('Найдем матрицу погрешностей H:')
    print('H=LU-PAQ: \n', L.dot(U) - P.dot(A.dot(Q)))
    print('||H||: ', find_norm(L.dot(U) - P.dot(A.dot(Q))))
    print('rank(A) = ', rank)
    print('Проверим систему на совместимость')
    y = solveLy(L, P.dot(b))
    print('y: \n', y)
    i = len(b) - 1
    count = 0
    while abs(y[i][0]) <= epsilon and i != -1:
        count += 1
        i -= 1
    if count >= len(y) - rank:
        print('Система совместна. Найдем частное решение: ')
        x = Q.dot(solveUx(U, y))
        print(x)
        print('Ax-b: ')
        print(A.dot(x)-b)
    else:
        print('Система несовместна')
else:
    swap_count = result[4]
    print('\nL: \n', L, '\nU: \n', U, '\nP: \n', P)
    print('\nПроверим, что LU = PA')
    print('LU: ', '\n', L.dot(U))
    print('PA: ', '\n', P.dot(A))
    print('\nL и U найдены с погрешностью из-за округлений')
    print('Найдем матрицу погрешностей H:')
    print('H=LU-PA: \n', L.dot(U)-P.dot(A))
    print('||H||: ', find_norm(L.dot(U)-P.dot(A)))

    print('\nПосчитаем определитель A:')
    print(np.linalg.det(A))
    print(find_determinant(U, swap_count))

    print('\nРешение СЛАУ:')
    resultLU = solveUx(U, solveLy(L, P.dot(b)))
    print(resultLU)
    print('Проверка Ax-b:')
    print(A.dot(resultLU) - b)

    print('\nНайдем матрицу, обратную А:')
    for i in range(A.shape[0]):
        e = np.zeros(b.shape)
        e[i] = 1
        curr = solveUx(U, solveLy(L, P.dot(e)))
        if i == 0:
            invA = curr
        else:
            invA = np.hstack((invA, curr))
    print(invA)
    print('A*A^(-1): \n', A.dot(invA))
    print('A^(-1)*A: \n', invA.dot(A))

    print('\nНайдем число обусловленности: ')
    cond = find_norm(A) * find_norm(invA)
    print(cond)