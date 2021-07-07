import math
from scipy import integrate
import numpy as np
import solveSLE

a = 0.8
b = 1.3
beta = 4 / 7
J = integrate.quad(lambda x: (5.7 * math.cos(2.5 * x) * math.exp(-4 * x / 7) + 4.4 * math.sin(4.3 * x) * math.exp(2 * x / 7) + 5) / (b-x) ** beta, a, b)[0]
print('точное значение интеграла: ', J)


j0 = lambda x: -7/3 * (1.3-x) ** (3/7)
j1 = lambda x: -(7 * (1.3-x)**(3/7) * (30*x+91)) / 300
j2 = lambda x: -(7 * (1.3-x)**(3/7) * (1500*x**2 + 2730*x + 8281)) / 25500
j3 = lambda x: -(7 * (1.3-x)**(3/7) * (85000*x**3 + 136500*x**2 + 248430*x + 753571)) / 2040000
j4 = lambda x: -(7 * (1.3-x)**(3/7) * (5100000*x**4 + 7735000*x**3 + 12421500*x**2 + 22607130*x + 68574961)) / 158100000
j5 = lambda x: -(7 * (1.3-x)**(3/7) * (316200000*x**5 + 464100000*x**4 + 703885000*x**3 + 1130356500*x**2 + 2057248830*x + 6240321451)) / 12015600000


def mu0(left, right):
    return j0(right) - j0(left)


def mu1(left, right):
    return j1(right) - j1(left)


def mu2(left, right):
    return j2(right) - j2(left)


def mu3(left, right):
    return j3(right) - j3(left)


def mu4(left, right):
    return j4(right) - j4(left)


def mu5(left, right):
    return j5(right) - j5(left)


def f(x):
    return 5.7 * math.cos(2.5 * x) * math.exp(-4 * x / 7) + 4.4 * math.sin(4.3 * x) * math.exp(2 * x / 7) + 5


def dddf(x):
    return math.exp(-4*x/7)*(75.1033*math.sin(2.5*x)-69.6311*math.exp(6*x/7)*math.sin(4.3*x)+60.0079*math.cos(2.5*x)-345.197*math.exp(6*x/7)*math.cos(4.3*x))


def cubicEquation(matrix):
    a = matrix[2, 0]
    b = matrix[1, 0]
    c = matrix[0, 0]
    Q = (a ** 2 - 3 * b) / 9
    R = (2 * a ** 3 - 9 * a * b + 27 * c) / 54
    S = Q ** 3 - R ** 2
    phi = math.acos(R / math.sqrt(Q ** 3)) / 3
    x1 = -2 * math.sqrt(Q) * math.cos(phi) - a / 3
    x2 = -2 * math.sqrt(Q) * math.cos(phi + 2 * math.pi / 3) - a / 3
    x3 = -2 * math.sqrt(Q) * math.cos(phi - 2 * math.pi / 3) - a / 3
    return np.array([x1, x2, x3])


def interpolationQF(x, left, right):
    nodes = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        nodes[i, :] = x ** i
    momentum = np.array([[mu0(left, right)], [mu1(left, right)], [mu2(left, right)]])
    A = solveSLE.solve(nodes, momentum)
    result = 0
    for i in range(3):
        result += A[i, 0] * f(x[i])
    return result


def compoundQF(h, L):
    z = np.zeros(L+1)
    result = 0
    for i in range(L+1):
        z[i] = a + i * h
    for i in range(1, L+1):
        x = np.array([z[i-1], (z[i-1] + z[i]) / 2, z[i]])
        result += interpolationQF(x, z[i-1], z[i])
    return result


def interpolationGauss(left, right):
    momentum = np.array([[mu0(left, right)], [mu1(left, right)], [mu2(left, right)], [mu3(left, right)], [mu4(left, right)], [mu5(left, right)]])
    matrix = np.array([[momentum[0, 0], momentum[1, 0], momentum[2, 0]],
              [momentum[1, 0], momentum[2, 0], momentum[3, 0]],
              [momentum[2, 0], momentum[3, 0], momentum[4, 0]]])
    b = np.array([[-momentum[3, 0]], [-momentum[4, 0]], [-momentum[5, 0]]])
    a = solveSLE.solve(matrix, b)
    x = cubicEquation(a)
    nodes = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        nodes[i, :] = x ** i
    A = solveSLE.solve(nodes, np.array([[mu0(left, right)], [mu1(left, right)], [mu2(left, right)]]))
    result = 0
    for i in range(3):
        result += A[i, 0] * f(x[i])
    return result


def compoundGauss(h, L):
    z = np.zeros(L+1)
    result = 0
    for i in range(L+1):
        z[i] = a + i * h
    for i in range(1, L+1):
        result += interpolationGauss(z[i - 1], z[i])
    return result


def richardson(intervals, integrals, m):
    l = len(intervals)
    if l < 2:
        return 1
    H = np.zeros((l, l), dtype=np.float64)
    J = np.zeros((l, 1), dtype=np.float64)
    for i in range(l):
        for j in range(l):
            if j == 0:
                H[i, j] = -1
            else:
                H[i, j] = intervals[i] ** (m+j-1)
        J[i, 0] = -integrals[i]
    C = solveSLE.solve(H, J)
    res = 0
    for i in range(1, l):
        res += C[i, 0] * intervals[l-1] ** (m+i-1)
    return res


x = np.array([a, (a+b)/2, b])
result = interpolationQF(x, a, b)
err = dddf(1.05) / math.factorial(3) * (-0.0022308125)
print('результат интерполяционной квадратурной формулы: ', result)
print('J-S: ', J - result, '\n')

eps = 1e-6
k = 0
cond = True
intervals = []
integrals = []
h = b - a
L = 2
m = 2
accuracy = 3
while cond:
    result = compoundQF(h, L ** k)
    intervals.append(h)
    integrals.append(result)

    if len(intervals) >= 3:
        S1 = integrals[len(integrals)-3]
        S2 = integrals[len(integrals)-2]
        S3 = integrals[len(integrals)-1]
        if (S3-S2) / (S2-S1) > 0:
            m = - math.log((S3-S2) / (S2-S1)) / math.log(L)
        if m < accuracy - 0.5 or m > accuracy + 1:
            m = accuracy

        cond = abs(richardson(intervals, integrals, m)) > eps

    h = h / L
    k += 1
print('результат составной КФ Ньютона-Котеса: ', result)
print('J-S: ', J - result)
print('k =', k, '\n')

R = (integrals[2] - integrals[1]) / (L ** (m) - 1)
h_opt = 0.95 * intervals[2] * (eps / abs(R)) ** (1/m)
h_opt = (b-a)/math.ceil((b-a)/h_opt)
print('оптимальный шаг: ', (b-a)/h_opt)
intervals = []
integrals = []
cond = True
k = 0
while cond:
    result = compoundQF(h_opt, int((b-a)/h_opt))
    intervals.append(h_opt)
    integrals.append(result)
    cond = abs(richardson(intervals, integrals, m)) > eps
    h_opt = h_opt / L
    k += 1
print('результат составной КФ Ньютона-Котеса при оптимальном шаге: ', result)
print('J-S: ', J - result)
print('k =', k, '\n')

result = interpolationGauss(a, b)
print('результат КФ Гаусса: ', result)
print('J-S: ', J-result, '\n')
eps = 1e-9
k = 0
cond = True
intervals = []
integrals = []
h = b - a
L = 2
m = 6
accuracy = 6
while cond:
    result = compoundGauss(h, L ** k)
    intervals.append(h)
    integrals.append(result)

    if len(intervals) >= 3:
        S1 = integrals[len(integrals)-3]
        S2 = integrals[len(integrals)-2]
        S3 = integrals[len(integrals)-1]
        if (S3-S2) / (S2-S1) > 0:
            m = - math.log((S3-S2) / (S2-S1)) / math.log(L)
        if m < accuracy - 0.5 or m > accuracy + 1:
            m = accuracy

        cond = abs(richardson(intervals, integrals, m)) > eps
    h = h / L
    k += 1
print('результат составной КФ Гаусса: ', result)
print('J-S: ', J - result)
print('k =', k, '\n')

R = (integrals[2] - integrals[1]) / (L ** (m) - 1)
h_opt = 0.95 * intervals[2] * (eps / abs(R)) ** (1/m)
h_opt = (b-a)/math.ceil((b-a)/h_opt)
print('оптимальный шаг: ', (b-a)/h_opt)
intervals = []
integrals = []
cond = True
k = 0
while cond:
    result = compoundGauss(h_opt, int((b-a)/h_opt))
    intervals.append(h_opt)
    integrals.append(result)
    cond = abs(richardson(intervals, integrals, m)) > eps
    h_opt = h_opt / L
    k += 1
print('результат составной КФ Гаусса при оптимальном шаге: ', result)
print('J-S: ', J - result)
print('k =', k, '\n')
