import pandas as pd
import sys
# sys.path.append('C:\\Users\\BMakunin\\SF\\mlops\\MLOps\\src')
sys.path.append('./../src')

from data.make_dataset import load_data, preprocess_data

# Проверяем функцию загрузки данных из csv
def test_load_data():
    # df = load_data('C:\\Users\\BMakunin\\SF\\mlops\\MLOps\\data\\raw\\UCI_Credit_Card.csv')
    df = load_data('./../data/raw/UCI_Credit_Card.csv')
    assert isinstance(df, pd.DataFrame), "Функция возвращает объект отличный от pd.DataFrame"
    assert len(df.columns) > 0, "Загружен пустой датасет"

# Проверяем функцию предобработки данных
def test_preprocess_data():
    # df = load_data('C:\\Users\\BMakunin\\SF\\mlops\\MLOps\\data\\raw\\UCI_Credit_Card.csv')
    df = load_data('./../data/raw/UCI_Credit_Card.csv')
    preprocessed_df = preprocess_data(df)
    assert isinstance(preprocessed_df, pd.DataFrame), "Функция возвращает объект отличный от pd.DataFrame"
    assert preprocessed_df.isnull().values.any() == False, "Присутствуют пропуски в предобработанных данных"

from models.pipeline import create_pipeline, optimize_hyperparameters, split_data

# Проверяем создание модели
def test_create_pipeline():
    trained_model = create_pipeline()
    assert hasattr(trained_model, 'fit'), "Модель создана некорректно. Отсутствует метод .fit"
    assert hasattr(trained_model, 'predict'), "Модель создана некорректно. Отсутствует метод .predict"

# Проверяем подбор гиперпараметров
def test_optimize_hyperparameters():
    # raw_path = 'C:\\Users\\BMakunin\\SF\\mlops\\MLOps\\data\\raw\\UCI_Credit_Card.csv'
    raw_path = './../data/raw/UCI_Credit_Card.csv'
    X_train, X_test, y_train, y_test = split_data(raw_path)
    pipe = create_pipeline()
    best_params, best_score = optimize_hyperparameters(pipe, X_train, y_train)
    assert len(best_params) == 2, "Некоторые гиперпараметры не рассчитаны"
    assert best_score > 0.5, "Слишком низкая точность модели"
