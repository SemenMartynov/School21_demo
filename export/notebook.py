# %%
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# Загрузка данных

# %%
data = pd.read_csv('customer_purchases.csv')

# %% [markdown]
# Обзор данных

# %%
data.describe() # Описательная статистика числовых признаков
data.nunique() # Количество уникальных значений для категориальных признаков
data.head()  # Первые 5 строк
data.info()  # Информация о датасете
data.describe()  # Статистика по числовым столбцам

# %% [markdown]
# 1. Топ-10 самых популярных видов покупок

# %%
data['Вид покупки'].value_counts().head(10).plot(kind='bar')
plt.title('Топ-10 самых популярных видов покупок')
plt.xlabel('Вид покупки')
plt.ylabel('Количество')
plt.show()

# %% [markdown]
# 2. Средняя сумма покупки по категориям клиентов

# %%
data.groupby('Категория клиента')['Сумма'].mean().plot(kind='bar')
plt.title('Средняя сумма покупки по категориям клиентов')
plt.xlabel('Категория клиента')
plt.ylabel('Средняя сумма покупки')
plt.show()

# %% [markdown]
# 3. Зависимость суммы покупки от возраста

# %%
plt.scatter(data['Возраст'], data['Сумма'])
plt.title('Зависимость суммы покупки от возраста')
plt.xlabel('Возраст')
plt.ylabel('Сумма покупки')
plt.show()

# %% [markdown]
# 4. Количество покупок по городам

# %%
data['Город'].value_counts().plot(kind='bar')
plt.title('Количество покупок по городам')
plt.xlabel('Город')
plt.ylabel('Количество покупок')
plt.show()

# %% [markdown]
# 5. Динамика покупок по времени

# %%
data['Дата операции'] = pd.to_datetime(data['Дата операции'])
data.groupby(data['Дата операции'].dt.month)['ID записи'].count().plot()
plt.title('Динамика покупок по месяцам')
plt.xlabel('Месяц')
plt.ylabel('Количество покупок')
plt.show()

# %% [markdown]
# 6. Сегментация клиентов по возрасту и сумме покупки

# %%
plt.scatter(data['Возраст'], data['Сумма'], c=data['Категория клиента'].astype('category').cat.codes)
plt.title('Сегментация клиентов')
plt.xlabel('Возраст')
plt.ylabel('Сумма покупки')
plt.show()

# %% [markdown]
# 7. Средний возраст клиентов по городам

# %%
avg_age_by_city = data.groupby('Город')['Возраст'].mean()
avg_age_by_city.plot(kind='bar', figsize=(10, 6))
plt.title('Средний возраст клиентов по городам')
plt.xlabel('Город')
plt.ylabel('Средний возраст')
plt.show()

# %% [markdown]
# 8. Количество покупок по месяцам

# %%
data['Месяц'] = pd.to_datetime(data['Дата операции']).dt.month
purchase_counts_by_month = data.groupby('Месяц')['ID записи'].count()
purchase_counts_by_month.plot(kind='line', figsize=(10, 6))
plt.title('Количество покупок по месяцам')
plt.xlabel('Месяц')
plt.ylabel('Количество покупок')
plt.show()

# %% [markdown]
# 9. Количество покупок по видам покупок и категориям клиентов

# %%
purchase_counts_by_type_and_category = data.groupby(['Вид покупки', 'Категория клиента'])['ID записи'].count().unstack()
purchase_counts_by_type_and_category.plot(kind='bar', figsize=(12, 6))
plt.title('Количество покупок по видам покупок и категориям клиентов')
plt.xlabel('Вид покупки')
plt.ylabel('Количество покупок')
plt.legend(title='Категория клиента')
plt.show()

# %% [markdown]
# 10. Корреляционная матрица числовых признаков

# %%
numeric_data = data[['Возраст', 'Сумма']]
corr_matrix = numeric_data.corr()
plt.figure(figsize=(6, 4))
plt.imshow(corr_matrix, cmap='coolwarm')
plt.colorbar()
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.title('Корреляционная матрица числовых признаков')
plt.show()


