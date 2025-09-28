# ml_module.py
import os
import zipfile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

import eda_module as eda
import decor_module as d

def load_data(file_path):
    """
    Загрузка данных из CSV файла.
    :param file_path: Путь к CSV файлу.
    :return: DataFrame с загруженными данными.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'File not found at {file_path}')

        if file_path.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zip_index:
                zip_files = zip_index.namelist()
                print(f'Files in zip {zip_files}')
                zip_file = zip_files[0]
                with zip_index.open(zip_file) as file:
                    return _load_by_extension(file_path, zip_file)
                    
        file_ext = os.path.splitext(file_path)[1].lower()

        return _load_by_extension(file_path, file_ext)

    except FileNotFoundError as e:
        print(f"File not found: {e}")

    except ValueError as e:
        print(f'Value error: {e}')

    except Exception as e:
        print(f'Undefined error: {e}')

    finally:
        print(f'Attempted data loading from file at {file_path}')


def _load_by_extension(file_path, file_ext):
    
    if file_ext == '.csv':
        return pd.read_csv(file_path)

    elif file_ext == '.json':
        return pd.read_json(file_path)

    elif file_ext == '.txt':
        return pd.read_txt(file_path, delimiter = '\t') # tab separation is assumed

    elif file_ext in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)

    elif file_ext == '.parquet':
        return pd.read_parquet(file_path)

    else:
        raise ValueError(f'Unsupported file {file_ext}')
    

def preprocess_data(df, drop_columns, target_column):
    """
    Предобработка данных: разделение на признаки и целевую переменную, масштабирование признаков.
    :param df: DataFrame с данными.
    :param target_column: Имя столбца с целевой переменной.
    :return: Обработанные признаки, целевая переменная, препроцессор.
    """

    if df is None:
        print("Данные не загружены. Вызовите метод load_data().")
        return
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date', inplace = False)
    df = df.drop(columns = drop_columns) 
    df = df.dropna() # remove nan values
    df = df.drop_duplicates() # remove duplicates

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(df)

    # imp = SimpleImputer(strategy = "most_frequent" )
    # imp.fit_transform(df)
    
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Определение числовых и категориальных признаков
    # numeric_features = ['', '', '']
    # categorical_features = ['', '', '']

    # # Создание препроцессора
    # numeric_transformer = StandardScaler()
    # categorical_transformer = OneHotEncoder(drop='first')

    # preprocessor = ColumnTransformer(
    #     transformers=[
    #         ('num', numeric_transformer, numeric_features),
    #         ('cat', categorical_transformer, categorical_features)
    #     ])

    # Применение препроцессора к данным
    X_processed = X
    print("Данные успешно предобработаны.")
    
    return X_processed, y

def train_model(X, y):
    """
    Обучение модели линейной регрессии.
    :param X: Признаки.
    :param y: Целевая переменная.
    :return: Обученная модель.
    """
    """
    Обучение модели на обучающих данных.
    """
    if X is None or y is None:
        print("Данные не загружены или не предобработаны.")
        return

    try:
        model = LinearRegression()
        model.fit(X, y)
        print("Модель успешно обучена.")
        return model
    except Exception as e:
        print(f"Ошибка при обучении модели: {e}")

def predict(model, X):
    """
    Предсказание на новых данных.
    :param model: Обученная модель.
    :param X: Признаки.
    :return: Предсказанные значения.
    """
    return model.predict(X)

def evaluate_model(y_true, y_pred):
    """
    Оценка модели с использованием метрик MSE и R^2.
    :param y_true: Истинные значения.
    :param y_pred: Предсказанные значения.
    :return: MSE, R^2.
    """
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2

