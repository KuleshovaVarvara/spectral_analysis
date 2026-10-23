import pandas as pd
import numpy as np

#Загрузка данных
df = pd.read_csv('variant_13.csv')
T = df['traffic'].values          # временной ряд
N = len(T)                        # 500
K = 50                            # сколько гармоник считаем

print(f"N = {N}, считаем k = 0..{K}")
print()

#Коэффициенты Фурье

# Постоянная составляющая: a0/2 = (1/N) * sum(T)
a0_half = np.sum(T) / N
print(f"a0/2 (среднее значение) = {a0_half:.4f}")
print()

# Массивы для коэффициентов (индексы 0..K)
a = np.zeros(K + 1)
b = np.zeros(K + 1)

# a[0] сохраняем как 2*(a0/2) = a0 — для единообразия формулы восстановления
a[0] = 2 * a0_half
b[0] = 0.0   # b0 всегда 0

# Основной двойной цикл — буквально по формулам из задания
for k in range(1, K + 1):
    sum_a = 0.0
    sum_b = 0.0
    for j in range(N):
        angle = 2.0 * np.pi * k * j / N
        sum_a += T[j] * np.cos(angle)
        sum_b += T[j] * np.sin(angle)
    a[k] = 2.0 / N * sum_a
    b[k] = 2.0 / N * sum_b

#Амплитуды A_k = sqrt(a_k^2 + b_k^2)
A = np.sqrt(a**2 + b**2)

#Для k=0 амплитуда — это просто |a0/2| = среднее
A[0] = a0_half

#Вывод таблицы
print(f"{'k':>3} | {'a_k':>10} | {'b_k':>10} | {'A_k':>10}")
print("-" * 42)
for k in range(K + 1):
    print(f"{k:>3} | {a[k]:>10.4f} | {b[k]:>10.4f} | {A[k]:>10.4f}")

#Сохраняем результаты в CSV для следующих шагов
result = pd.DataFrame({
    'k':   np.arange(K + 1),
    'a_k': a,
    'b_k': b,
    'A_k': A
})
result.to_csv('fourier_coeffs.csv', index=False)
print()
print("Коэффициенты сохранены в fourier_coeffs.csv")

#Топ-10 гармоник по амплитуде (без k=0)
print()
print("Топ-10 гармоник по амплитуде (k >= 1):")
order = np.argsort(A[1:])[::-1] + 1     # сортируем без нулевой
for k in order[:10]:
    print(f"  k = {k:>2}  A_k = {A[k]:.4f}")