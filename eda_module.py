#EDA module

# Импорт стандартных модулей
import sys
import collections
from typing import List, Tuple

# Импорт сторонних библиотек
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns

import decor_module as d

#----------------------------------Show Types and Stats------------------------------------------------
@d.data_descr_decorator
def view_data(df):

    if df is None:
        print("Данные не загружены. Вызовите метод load_data().")
        return
    print(df.info())
    print(df.describe())
    print(df.head())
    print(df.dtypes)
    return df


#-----------------------------------Histograms---------------------------------------------------------

@d.hist_decorator
def plot_histogram(df):
    
    if df is None:
        print("Данные не загружены. Вызовите метод load_data().")
        return
    # Установка стиля Seaborn для красивых графиков
    sns.set(style="whitegrid")
    
    # Создание гистограмм для каждой числовой переменной
    df.hist(bins=20, figsize=(15, 10), color='skyblue', edgecolor='black')
    
    # Добавление названий для каждого графика и осей
    for ax in plt.gcf().get_axes():
        ax.set_xlabel('Значение')
        ax.set_ylabel('Частота')
        ax.set_title(ax.get_title())
        #ax.set_title(ax.get_title().replace('wine_class', 'Класс вина'))
    
    # Регулировка макета для предотвращения наложения подписей
    plt.tight_layout()
    
    # Показать график
    plt.show()


#-------------------------------------Heatmaps----------------------------------------------------------
@d.heatmap_decorator
def plot_heatmap(df):
    
    if df is None:
        print("Данные не загружены. Вызовите метод load_data().")
        return
    # Установка стиля Seaborn
    sns.set(style="white")
    
    # Расчет корреляционной матрицы только для числовых данных
    numeric_df = df.select_dtypes(include=[np.number])  # Исключаем нечисловые столбцы
    corr = numeric_df.corr()
    
    # Маска для отображения только нижней треугольной части матрицы (опционально)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Настройка цветовой палитры
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # Создание тепловой карты
    plt.figure(figsize=(30, 16))
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
    
    # Добавление заголовка
    plt.title('Тепловая карта корреляций', fontsize=20)
    
    # Показать график
    plt.show()


#------------------------------------WhiskerBox-----------------------------------------------------------
@d.whisker_box_decorator
def plot_whisker_box(df):

    if df is None:
        print("Данные не загружены. Вызовите метод load_data().")
        return
    # Установка стиля Seaborn
    sns.set(style="whitegrid")
    
    # Предполагаем, что df — это ваш DataFrame
    # Создаем ящики с усами для каждой колонки в DataFrame
    plt.figure(figsize=(12, 50))
    
    # Перебираем каждый числовой столбец и создаем для него ящик с усами
    for index, column in enumerate(df.select_dtypes(include=[np.number]).columns):
        plt.subplot((len(df.columns) // 3) + 1, 3, index + 1)
        sns.boxplot(y=df[column])
    
    plt.tight_layout()
    plt.show()
