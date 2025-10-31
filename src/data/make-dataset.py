import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import os
from nbconvert import ScriptExporter


# In[23]:



raw_path = r"C:\Users\BMakunin\SF\mlops\MLOps\data\raw\\"
processed_path = r"C:\Users\BMakunin\SF\mlops\MLOps\data\processed\\"


# In[24]:


# Создаем функцию для загрузки датасета
def load_data(data_path=raw_path+'UCI_Credit_Card.csv'):
    df = pd.read_csv(data_path)
    return df


# In[25]:


# Посмотрим на наш датасет
df = load_data()
print(f'Размер таблицы: {df.shape}')
print('Первые строки:')
print(df.head())
print('Типы данных:')
print(df.dtypes)


# In[15]:


# Создадим функцию, которая будет производить разведывательный анализ датасета
def explore_data(df):
    # Находим количсество пропусков в данных
    print('Количество пропусков:')
    print(df.isna().sum())

    # Оцениваем процентное распределение целевой переменной
    target_distribution = df['default.payment.next.month'].value_counts(normalize=True)*100
    print('\nРаспределение целевой переменной (%):')
    print(target_distribution)

    # Выводим статистику по числовым признакам
    numeric_columns = df.select_dtypes(include=['float', 'int']).columns
    print('\nСтатистика по числовым признакам:')
    print(df[numeric_columns].describe())

    # Выведем значения категориальных признаков
    categorical_columns = df.select_dtypes(exclude=['float', 'int']).columns
    if len(categorical_columns) > 0:
        print('\nКатегоричные признаки:')
        for col in categorical_columns:
            print(f'{col}: {df[col].unique()}')


# In[16]:


explore_data(df)


# In[17]:


# Создадим функцию, которая будет предобрабатывать данные
def preprocess_data(df):
    # Удаляем столбец ID
    df.drop(columns=['ID'], inplace=True)

    # Производим биннинг возраста
    bins = [20, 30, 40, 50, float('inf')]
    labels = ['Молодые', 'Средний возраст', 'Взрослые', 'Старшие']
    df['AGE_GROUP'] = pd.cut(df['AGE'], bins=bins, labels=labels)

    # Выполняем OHE-обработку категориальных признаков
    cat_cols = ['SEX', 'MARRIAGE', 'EDUCATION']
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Нормализуем числовые признаки
    scaler = MinMaxScaler()
    num_cols = ['LIMIT_BAL'] + [f'BILL_AMT{i}' for i in range(1, 7)] + [f'PAY_AMT{i}' for i in range(1, 7)]
    df_scaled = scaler.fit_transform(df_encoded[num_cols])
    df_encoded[num_cols] = df_scaled

    return df_encoded


# In[26]:


# Обрабатываем данные и сохраняем датасет
df_processed = preprocess_data(df)
output_path = Path(processed_path+'preprocessed_data.csv')
df_processed.to_csv(output_path, index=False)
print(f'\nДанные успешно сохранены в {output_path}.')


# In[28]:


