import pandas as pd
import matplotlib.pyplot as plt

#Загрузка данных
df = pd.read_csv('variant_13.csv')

print("Первые 5 строк:")
print(df.head())
print()
print("Размер таблицы:", df.shape)
print("Колонки:", list(df.columns))
print()
print("Средний трафик:", df['traffic'].mean())
print("Минимум:", df['traffic'].min())
print("Максимум:", df['traffic'].max())

#График
plt.figure(figsize=(14, 5))
plt.plot(df['index'], df['traffic'], linewidth=0.8, color='steelblue')

plt.title('Интенсивность сетевого трафика (variant_13)')
plt.xlabel('index (время)')
plt.ylabel('traffic (пакетов/с)')
plt.grid(True, alpha=0.3)
plt.tight_layout()

#Сохранить картинку для отчёта
plt.savefig('step1_traffic.png', dpi=150)

#Показать на экране
plt.show()