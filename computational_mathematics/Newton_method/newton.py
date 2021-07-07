import numpy as np
import math
import lu
import time


def find_norm(x):
    return max(abs(x[:, 0]))


def F(x):
    return np.array([[
    math.cos(x[1, 0] * x[0, 0]) - math.exp(-3 * x[2, 0]) + x[3, 0] * x[4, 0] ** 2 - x[5, 0] - math.sinh(2 * x[7, 0]) * x[8, 0] + 2 * x[9, 0] + 2.000433974165385440],
    [math.sin(x[1, 0] * x[0, 0]) + x[2, 0] * x[8, 0] * x[6, 0] - math.exp(-x[9, 0] + x[5, 0]) + 3 * x[4, 0] ** 2 - x[5, 0] * (x[7, 0] + 1) + 10.886272036407019994],
    [x[0, 0] - x[1, 0] + x[2, 0] - x[3, 0] + x[4, 0] - x[5, 0] + x[6, 0] - x[7, 0] + x[8, 0] - x[9, 0] - 3.1361904761904761904],
    [2 * math.cos(-x[8, 0] + x[3, 0]) + x[4, 0] / (x[2, 0] + x[0, 0]) - math.sin(x[1, 0] ** 2) + math.cos(x[6, 0] * x[9, 0]) ** 2 - x[7, 0] - 0.1707472705022304757],
    [math.sin(x[4, 0]) + 2 * x[7, 0] * (x[2, 0] + x[0, 0]) - math.exp(-x[6, 0] * (-x[9, 0] + x[5, 0])) + 2 * math.cos(x[1, 0]) - 1.0 / (-x[8, 0] + x[3, 0]) - 0.3685896273101277862],
    [math.exp(x[0, 0] - x[3, 0] - x[8, 0]) + x[4, 0] ** 2 / x[7, 0] + math.cos(3 * x[9, 0] * x[1, 0]) / 2 - x[5, 0] * x[2, 0] + 2.0491086016771875115],
    [x[1, 0] ** 3 * x[6, 0] - math.sin(x[9, 0] / x[4, 0] + x[7, 0]) + (x[0, 0] - x[5, 0]) * math.cos(x[3, 0]) + x[2, 0] - 0.7380430076202798014],
    [x[4, 0] * (x[0, 0] - 2 * x[5, 0]) ** 2 - 2 * math.sin(-x[8, 0] + x[2, 0]) + 0.15e1 * x[3, 0] - math.exp(x[1, 0] * x[6, 0] + x[9, 0]) + 3.5668321989693809040],
    [7 / x[5, 0] + math.exp(x[4, 0] + x[3, 0]) - 2 * x[1, 0] * x[7, 0] * x[9, 0] * x[6, 0] + 3 * x[8, 0] - 3 * x[0, 0] - 8.4394734508383257499],
    [x[9, 0] * x[0, 0] + x[8, 0] * x[1, 0] - x[7, 0] * x[2, 0] + math.sin(x[3, 0] + x[4, 0] + x[5, 0]) * x[6, 0] - 0.78238095238095238096]])


def J(x):
    return np.array([[-x[1, 0] * math.sin(x[1, 0] * x[0, 0]), -x[0, 0] * math.sin(x[1, 0] * x[0, 0]), 3 * math.exp(-3 * x[2, 0]), x[4, 0] ** 2, 2 * x[3, 0] * x[4, 0],
                -1, 0, -2 * math.cosh(2 * x[7, 0]) * x[8, 0], -math.sinh(2 * x[7, 0]), 2],
               [x[1, 0] * math.cos(x[1, 0] * x[0, 0]), x[0, 0] * math.cos(x[1, 0] * x[0 ,0]), x[8, 0] * x[6, 0], 0, 6 * x[4, 0],
                -math.exp(-x[9, 0] + x[5, 0]) - x[7, 0] - 1, x[2, 0] * x[8, 0], -x[5, 0], x[2, 0] * x[6, 0], math.exp(-x[9, 0] + x[5, 0])],
               [1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
               [-x[4, 0] / (x[2, 0] + x[0, 0]) ** 2, -2 * x[1, 0] * math.cos(x[1, 0] ** 2), -x[4, 0] / (x[2, 0] + x[0, 0]) ** 2, -2 * math.sin(-x[8, 0] + x[3, 0]),
                1.0 / (x[2, 0] + x[0, 0]), 0, -2 * math.cos(x[6, 0] * x[9, 0]) * x[9, 0] * math.sin(x[6, 0] * x[9, 0]), -1,
                2 * math.sin(-x[8, 0] + x[3, 0]), -2 * math.cos(x[6, 0] * x[9, 0]) * x[6, 0] * math.sin(x[6, 0] * x[9, 0])],
               [2 * x[7, 0], -2 * math.sin(x[1, 0]), 2 * x[7, 0], 1.0 / (-x[8, 0] + x[3, 0]) ** 2, math.cos(x[4, 0]),
                x[6, 0] * math.exp(-x[6, 0] * (-x[9, 0] + x[5, 0])), -(x[9, 0] - x[5, 0]) * math.exp(-x[6, 0] * (-x[9, 0] + x[5, 0])), 2 * x[2, 0] + 2 * x[0, 0],
                -1.0 / (-x[8, 0] + x[3, 0]) ** 2, -x[6, 0] * math.exp(-x[6, 0] * (-x[9, 0] + x[5, 0]))],
               [math.exp(x[0, 0] - x[3, 0] - x[8, 0]), -1.5 * x[9, 0] * math.sin(3 * x[9, 0] * x[1, 0]), -x[5, 0], -math.exp(x[0, 0] - x[3, 0] - x[8, 0]),
                2 * x[4, 0] / x[7, 0], -x[2, 0], 0, -x[4, 0] ** 2 / x[7, 0] ** 2, -math.exp(x[0, 0] - x[3, 0] - x[8, 0]), -1.5 * x[1, 0] * math.sin(3 * x[9, 0] * x[1, 0])],
               [math.cos(x[3, 0]), 3 * x[1, 0] ** 2 * x[6, 0], 1, -(x[0, 0] - x[5, 0]) * math.sin(x[3, 0]), x[9, 0] / x[4, 0] ** 2 * math.cos(x[9, 0] / x[4, 0] + x[7, 0]),
                -math.cos(x[3, 0]), x[1, 0] ** 3, -math.cos(x[9, 0] / x[4, 0] + x[7, 0]), 0, -1.0 / x[4, 0] * math.cos(x[9, 0] / x[4, 0] + x[7, 0])],
               [2 * x[4, 0] * (x[0, 0] - 2 * x[5, 0]), -x[6, 0] * math.exp(x[1, 0] * x[6, 0] + x[9, 0]), -2 * math.cos(-x[8, 0] + x[2, 0]), 1.5,
               (x[0, 0] - 2 * x[5, 0]) ** 2, -4 * x[4, 0] * (x[0, 0] - 2 * x[5, 0]), -x[1, 0] * math.exp(x[1, 0] * x[6, 0] + x[9, 0]), 0, 2 * math.cos(-x[8, 0] + x[2, 0]),
                -math.exp(x[1, 0] * x[6, 0] + x[9, 0])],
               [-3, -2 * x[7, 0] * x[9, 0] * x[6, 0], 0, math.exp(x[4, 0] + x[3, 0]), math.exp(x[4, 0] + x[3, 0]),
                -7.0 / x[5, 0] ** 2, -2 * x[1, 0] * x[7, 0] * x[9, 0], -2 * x[1, 0] * x[9, 0] * x[6, 0], 3, -2 * x[1, 0] * x[7, 0] * x[6, 0]],
               [x[9, 0], x[8, 0], -x[7, 0], math.cos(x[3, 0] + x[4, 0] + x[5, 0]) * x[6, 0], math.cos(x[3, 0] + x[4, 0] + x[5, 0]) * x[6, 0],
                math.cos(x[3, 0] + x[4, 0] + x[5, 0]) * x[6, 0], math.sin(x[3, 0] + x[4, 0] + x[5, 0]), -x[2, 0], x[1, 0], x[0, 0]]])


def f(x):
    return x - math.sin(x) - 0.25


def df(x):
    return 1 - math.cos(x)


def d2f(x):
    return math.sin(x)


def newton_scalar(eps, a, b):
    x_prev = b if f(b) * d2f(b) > 0 else a
    x = x_prev - f(x_prev) / df(x_prev)
    k = 0
    while abs(x-x_prev) > eps:
        x_prev = x
        x = x_prev - f(x_prev) / df(x_prev)
        k += 1
    print('Количество итераций в методе Ньютона: ', k)
    return x


def newton_scalar_modified(eps, a, b):
    x_prev = b if f(b) * d2f(b) > 0 else a
    df0 = df(x_prev)
    x = x_prev - f(x_prev) / df0
    k = 0
    while abs(x - x_prev) > eps:
        x_prev = x
        x = x_prev - f(x_prev) / df0
        k += 1
    print('Количество итераций в модифицированном методе Ньютона: ', k)
    return x


def newton(eps, x_prev):
    max_iter = 100
    start = time.time()
    count_solve = 0
    count_lu = 0

    cond = True
    k = 0
    while cond:
        lu_result = lu.LU(J(x_prev), count_lu)
        l = lu_result[0]
        u = lu_result[1]
        p = lu_result[2]
        count_lu = lu_result[3]

        solve_result = lu.solveLU(l, u, p.dot(-F(x_prev)), count_solve)
        delta = solve_result[0]
        count_solve = solve_result[1]

        x = delta + x_prev
        k += 1

        if k == 2:
            x = newton_modified(eps, x, count_lu, count_solve, k, start)
            return x

        if k >= max_iter:
            break

        cond = find_norm(x - x_prev) > eps
        x_prev = x

    stop = time.time() - start
    print('Время выполнения: ', stop)
    print('Количество итераций: ', k)
    print('Количество арифметических операций в LU разложении: ', count_lu)
    print('Количество арифметических операций в решении СЛАУ: ', count_solve)
    return x


def newton_modified(eps, x_prev, count_lu=0, count_solve=0, k=0, start=time.time()):
    max_iter = 100
    lu_result = lu.LU(J(x_prev), count_lu)
    l = lu_result[0]
    u = lu_result[1]
    p = lu_result[2]
    count_lu = lu_result[3]

    cond = True
    while cond:
        solve_result = lu.solveLU(l, u, p.dot(-F(x_prev)), count_solve)
        delta = solve_result[0]
        count_solve = solve_result[1]

        x = delta + x_prev
        k += 1
        cond = find_norm(x-x_prev) > eps
        x_prev = x
        if k == max_iter:
            print('метод разошелся')
            break

    stop = time.time() - start
    print('Время выполнения: ', stop)
    print('Количество итераций: ', k)
    print('Количество арифметических операций в LU разложении: ', count_lu)
    print('Количество арифметических операций в решении СЛАУ: ', count_solve)
    return x


def hybrid(eps, x_prev, recount):
    start = time.time()
    count_solve = 0
    count_lu = 0

    cond = True
    k = 0
    while cond:
        if k % recount == 0:
            lu_result = lu.LU(J(x_prev), count_lu)
            l = lu_result[0]
            u = lu_result[1]
            p = lu_result[2]
            count_lu = lu_result[3]

        solve_result = lu.solveLU(l, u, p.dot(-F(x_prev)), count_solve)
        delta = solve_result[0]
        count_solve = solve_result[1]

        x = delta + x_prev
        k += 1

        cond = find_norm(x - x_prev) > eps
        x_prev = x

    stop = time.time() - start
    print('Время выполнения: ', stop)
    print('Количество итераций: ', k)
    print('Количество арифметических операций в LU разложении: ', count_lu)
    print('Количество арифметических операций в решении СЛАУ: ', count_solve)
    return x


# Вычисление для уравнения x - sin(x) = 0.25
eps = 1e-6
a = -1
b = 3
x1 = newton_scalar(eps, a, b)
print('Решение обычным методом Ньютона: ', x1)
print('Проверка: ', f(x1), '\n')
x2 = newton_scalar_modified(eps, a, b)
print('Решение модифицированным методом Ньютона: ', x2)
print('Проверка: ', f(x2), '\n')

# Нелинейные уравнения
x0 = np.array([[0.5], [0.5], [1.5], [-1.0], [-0.2], [1.5], [0.5], [-0.5], [1.5], [-1.5]])
x3 = newton(eps, x0)
print('Решение системы уравнений методом Ньютона: \n', x3)
print('Проверка подстановкой в F(x): \n', F(x3), '\n')
# x4 = newton_modified(eps, x0)
# print('Решение системы уравнений модифицированным методом Ньютона: \n', x4)
# print('Проверка подстановкой в F(x): \n', F(x4), '\n')
x5 = hybrid(eps, x0, recount=3)
print('Решение системы уравнений гибридным методом: \n', x5)
print('Проверка подстановкой в F(x): \n', F(x5), '\n')
