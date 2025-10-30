#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import os
from nbconvert import ScriptExporter


# In[23]:


# raw_path = './../data/raw/'                   # �������
raw_path = './../../data/raw/'                # ������
# processed_path = './../data/processed/'       # �������
processed_path = './../../data/processed/'    # ������


# In[24]:


# ������� ������� ��� �������� ��������
def load_data(data_path=raw_path+'UCI_Credit_Card.csv'):
    """
    ������� �������� ������ �� csv �����.
    """
    df = pd.read_csv(data_path)
    return df


# In[25]:


# ��������� �� ��� �������
if __name__ == "__main__":
    df = load_data()
    print(f'������ �������: {df.shape}')
    print('������ ������:')
    print(df.head())
    print('���� ������:')
    print(df.dtypes)


# In[15]:


# �������� �������, ������� ����� ����������� ���������������� ������ ��������
def explore_data(df):
    """
    ������� ��� ������� ������
    """
    # ������� ����������� ��������� � ������
    print('���������� ���������:')
    print(df.isna().sum())

    # ��������� ���������� ������������� ������� ����������
    target_distribution = df['default.payment.next.month'].value_counts(normalize=True)*100
    print('\n������������� ������� ���������� (%):')
    print(target_distribution)

    # ������� ���������� �� �������� ���������
    numeric_columns = df.select_dtypes(include=['float', 'int']).columns
    print('\n���������� �� �������� ���������:')
    print(df[numeric_columns].describe())

    # ������� �������� �������������� ���������
    categorical_columns = df.select_dtypes(exclude=['float', 'int']).columns
    if len(categorical_columns) > 0:
        print('\n������������ ��������:')
        for col in categorical_columns:
            print(f'{col}: {df[col].unique()}')


# In[16]:


explore_data(df)


# In[17]:


# �������� �������, ������� ����� ���������������� ������
def preprocess_data(df):
    """
    ��������������� ��������� ������
    """
    # ������� ������� ID
    df.drop(columns=["ID"], inplace=True)

    # ���������� ������� ��������
    bins = [20, 30, 40, 50, float('inf')]
    labels = ['�������', '������� �������', '��������', '�������']
    df['AGE_GROUP'] = pd.cut(df['AGE'], bins=bins, labels=labels)

    # ��������� OHE-��������� �������������� ���������
    cat_cols = ['SEX', 'MARRIAGE', 'EDUCATION']
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # ����������� �������� ��������
    scaler = MinMaxScaler()
    num_cols = ['LIMIT_BAL'] + [f'BILL_AMT{i}' for i in range(1, 7)] + [f'PAY_AMT{i}' for i in range(1, 7)]
    df_scaled = scaler.fit_transform(df_encoded[num_cols])
    df_encoded[num_cols] = df_scaled

    return df_encoded


# In[26]:


# ������������ ������ � ��������� �������
if __name__ == "__main__":
    df_processed = preprocess_data(df)
    output_path = Path(processed_path+'preprocessed_data.csv')
    df_processed.to_csv(output_path, index=False)
    print(f'\n������ ������� ��������� � {output_path}.')


# In[28]:


