import math


# формулы приведения для косинуса
def my_cos(x):
    while abs(x) > math.pi:
        x -= 2 * math.pi
    if 0 <= abs(x) <= (math.pi / 4):
        return my_c(abs(x))
    if (math.pi / 4) < abs(x) <= (math.pi / 2):
        return my_s(math.pi / 2 - abs(x))
    if (math.pi / 2) < abs(x) <= (3 * math.pi / 4):
        return -my_s(-math.pi / 2 + abs(x))
    if (3 * math.pi / 4) < abs(x) <= math.pi:
        return -my_c(math.pi - abs(x))


# вычисляем косинус
def my_c(x):
    last_num = 1
    k = 0
    cos = 0
    while abs(last_num) >= 10 ** (-6):
        last_num = (-1) ** k * x ** (2 * k) / math.factorial(2 * k)
        k += 1
        cos += last_num
    return cos


# вычисляем синус
def my_s(x):
    last_num = 1
    k = 0
    sin = 0
    while abs(last_num) >= 10 ** (-6):
        last_num = (-1) ** k * x ** (2 * k + 1) / math.factorial(2 * k + 1)
        k += 1
        sin += last_num
    return sin


# вычисляем arctg
def my_arctg(x):
    last_num = 1
    k = 0
    # проверяем x на соответствие условиям
    if abs(x) < 1:
        arctg = 0
        while abs(last_num) >= 10**(-6):
            last_num = (-1)**k * x**(2 * k + 1) / (2 * k + 1)
            k += 1
            arctg += last_num
    else:
        if x > 0:
            arctg = math.pi / 2
        elif x == 0:
            arctg = 0
        else:
            arctg = -math.pi / 2
        while abs(last_num) >= 10**(-6):
            last_num = (-1)**k * x**(-2 * k - 1) / (2 * k + 1)
            k += 1
            arctg -= last_num
    return arctg


# вычисляем корень
def my_sqrt(x):
    prev = 0.0
    sqrt = 1.0
    while abs(sqrt - prev) > 10**(-6):
        prev = sqrt
        sqrt = 0.5 * (prev + x / prev)
    return sqrt


# выводим результаты
x = 0.01
print('x', end='\t\t\t')
print('f_exact', end='\t\t\t\t')
print('f_approx', end='\t\t\t')
print('error', end='\n')
while x <= 0.06:
    print(x, end='\t')
    print(math.sqrt(2 * x + 0.4) * math.atan(math.cos(3 * x + 1)), end='\t')
    print(my_sqrt(2 * x + 0.4) * my_arctg(my_cos(3 * x + 1)), end='\t')
    print(abs(math.sqrt(2 * x + 0.4) * math.atan(math.cos(3 * x + 1))-my_sqrt(2 * x + 0.4) * my_arctg(my_cos(3 * x + 1))))
    x += 0.005
    x = round(x, 3)
