import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Загрузка исходного ряда
df = pd.read_csv('variant_13.csv')
T = df['traffic'].values
t = df['index'].values
N = len(T)

#Загрузка коэффициентов Фурье со Шага 2
coeffs = pd.read_csv('fourier_coeffs.csv')
a = coeffs['a_k'].values
b = coeffs['b_k'].values

# Функция восстановления ряда по первым K гармоникам
#   T(t) ≈ a0/2 + Σ_{k=1..K} [ a_k·cos(2πkt/N) + b_k·sin(2πkt/N) ]
def reconstruct(a, b, K, N):
    t_arr = np.arange(N)
    result = np.full(N, a[0] / 2.0)          # a0/2 — средний уровень
    for k in range(1, K + 1):
        angle = 2 * np.pi * k * t_arr / N
        result += a[k] * np.cos(angle) + b[k] * np.sin(angle)
    return result

# Восстанавливаем для K = 2 и K = 20
T_K2  = reconstruct(a, b, 2,  N)
T_K20 = reconstruct(a, b, 20, N)

# График сравнения
plt.figure(figsize=(14, 6))

plt.plot(t, T,     color='lightgray', linewidth=1.2, label='Исходный ряд')
plt.plot(t, T_K2,  color='crimson',   linewidth=1.5, label='Приближение K = 2')
plt.plot(t, T_K20, color='navy',      linewidth=1.2, label='Приближение K = 20')

plt.title('Тригонометрический многочлен: K = 2 и K = 20 против исходного ряда')
plt.xlabel('index (время)')
plt.ylabel('traffic (пакетов/с)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('step5_approx.png', dpi=150)
plt.show()

# Отдельный крупный график для K=2 (чтобы было видно, что он не ловит)
plt.figure(figsize=(14, 5))
plt.plot(t, T,    color='lightgray', linewidth=1.2, label='Исходный ряд')
plt.plot(t, T_K2, color='crimson',   linewidth=2.0, label='K = 2')
plt.title('Приближение K = 2 (недостаточное)')
plt.xlabel('index')
plt.ylabel('traffic')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('step5_K2.png', dpi=150)
plt.show()

# Отдельный крупный график для K=20
plt.figure(figsize=(14, 5))
plt.plot(t, T,     color='lightgray', linewidth=1.2, label='Исходный ряд')
plt.plot(t, T_K20, color='navy',      linewidth=1.2, label='K = 20')
plt.title('Приближение K = 20 (хорошее)')
plt.xlabel('index')
plt.ylabel('traffic')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('step5_K20.png', dpi=150)
plt.show()

# Метрики качества приближения
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

mse2,  rmse2  = mse(T, T_K2),  rmse(T, T_K2)
mse20, rmse20 = mse(T, T_K20), rmse(T, T_K20)

print("=" * 55)
print("Качество приближения")
print("=" * 55)
print(f"K = 2   :  MSE = {mse2:>10.2f}   RMSE = {rmse2:>8.2f}")
print(f"K = 20  :  MSE = {mse20:>10.2f}   RMSE = {rmse20:>8.2f}")
print()

# Доля объяснённой дисперсии
var_T = np.var(T)
print(f"Дисперсия исходного ряда:      {var_T:.2f}")
print(f"Остаток при K=2 :              {mse2:.2f}  ({100*mse2/var_T:.1f}% дисперсии)")
print(f"Остаток при K=20:              {mse20:.2f}  ({100*mse20/var_T:.1f}% дисперсии)")
print()

#Проверим, действительно ли K=7 восстанавливает всё
T_K7 = reconstruct(a, b, 7, N)
mse7 = mse(T, T_K7)
print(f"Для сравнения, K = 7  : MSE = {mse7:.2f}  ({100*mse7/var_T:.1f}% дисперсии)")