import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Загрузка коэффициентов, посчитанных на Шаге 2
coeffs = pd.read_csv('fourier_coeffs.csv')
k_vals = coeffs['k'].values
A      = coeffs['A_k'].values
a      = coeffs['a_k'].values
b      = coeffs['b_k'].values


# График 1. Полный спектр (k = 0..50), включая DC-составляющую
plt.figure(figsize=(14, 5))
plt.bar(k_vals, A, width=0.8, color='steelblue', edgecolor='black', linewidth=0.3)

plt.title('Амплитудный спектр A_k (k = 0..50)')
plt.xlabel('k (номер гармоники)')
plt.ylabel('A_k')
plt.xticks(range(0, 51, 5))
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('step3_spectrum_full.png', dpi=150)
plt.show()

# График 2. Спектр без DC (k = 1..50), чтобы увидеть слабые пики
mask = k_vals >= 1
plt.figure(figsize=(14, 5))
plt.bar(k_vals[mask], A[mask], width=0.8,
        color='indianred', edgecolor='black', linewidth=0.3)

plt.title('Амплитудный спектр A_k без постоянной составляющей (k = 1..50)')
plt.xlabel('k (номер гармоники)')
plt.ylabel('A_k')
plt.xticks(range(1, 51, 2))

# Линия шумового уровня — среднее по всем k, кроме доминирующего пика
noise_mask = (k_vals >= 1) & (k_vals != 7)
noise_level = A[noise_mask].mean()
plt.axhline(noise_level, color='black', linestyle='--', linewidth=1,
            label=f'Средний шумовой уровень = {noise_level:.2f}')

# Отметим главный пик
A7 = A[k_vals == 7][0]
plt.annotate(f'k=7\nA={A7:.1f}',
             xy=(7, A7), xytext=(12, A7*0.9),
             arrowprops=dict(arrowstyle='->', color='black'),
             fontsize=11, ha='left')

plt.legend()
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('step3_spectrum_noDC.png', dpi=150)
plt.show()

# Числовая сводка
print("=" * 55)
print("Сводка по спектру")
print("=" * 55)
print(f"Средний уровень (k=0):    A_0 = {A[0]:.2f}")
print(f"Главный пик:              k = 7,  A_7 = {A[7]:.2f}")
print(f"Уровень шума (k≠0, k≠7):  {noise_level:.2f}")
print(f"Превышение пика над шумом: {A7/noise_level:.1f}×")
print()

# Все гармоники, у которых амплитуда > 3× шумового уровня
threshold = 3 * noise_level
print(f"Гармоники с A_k > 3 × шум ({threshold:.2f}):")
for k, ak in zip(k_vals, A):
    if k >= 1 and ak > threshold:
        print(f"  k = {k:>2}   A_k = {ak:.4f}")

# Периоды
print()
print("Интерпретация периодов:")
for k in [1, 2, 7, 14, 21, 25]:
    if k <= 50:
        T_period = 500 / k
        print(f"  k = {k:>2}  →  период T = 500/{k} = {T_period:.2f} точек")