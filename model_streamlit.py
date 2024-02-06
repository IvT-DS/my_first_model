# Импортируем библиотеки
import pandas as pd

# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tqdm import tqdm
import joblib  # Библиотека для сохранения пайплайнов/моделей
import streamlit as st

# import warnings
# warnings.filterwarnings("ignore")

# Важная настройка для корректной настройки pipeline!
import sklearn

sklearn.set_config(transform_output="pandas")

# Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline, make_pipeline
# from sklearn.base import BaseEstimator, TransformerMixin

# Preprocessing
# from sklearn.svm import SVC
# from sklearn.impute import SimpleImputer
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler, OrdinalEncoder, TargetEncoder
# from sklearn.model_selection import GridSearchCV, KFold

# for model learning
# from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score

# models
# from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
# from catboost import CatBoostClassifier, CatBoostRegressor
# import lightgbm as lgb
# from lightgbm import LGBMClassifier
# from xgboost import XGBClassifier

# Metrics
# from sklearn.metrics import accuracy_score, make_scorer


# tunning hyperparamters model
# import optuna

# from tabulate import tabulate

# Загрузим датасет
# df = pd.read_csv('aux/heart.csv')

# # Разделим датафрейм на числовые и категориальные данные.

# num_features = df.select_dtypes(exclude='object')
# cat_features = df.select_dtypes(include='object')

# # Разделим сразу на features и target. Target - столбец "HeartDisease"
# X, y = df.drop('HeartDisease', axis=1), df['HeartDisease']

# X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# # импортируем препроцессор

# preprocessor = joblib.load("aux/preprocessor.pkl")

# # Voting essemble model
# lr = LogisticRegression()
# dt = DecisionTreeClassifier(max_depth=5, criterion='gini', random_state=42)
# knn = KNeighborsClassifier(n_neighbors=10, p=2, weights='distance')

# vc = VotingClassifier(
#     [
#         ('LogReg', lr),
#         ('DecisionTree', dt),
#         ('KNeighbors', knn)
#     ]
# )

# ml_pipeline_VC = Pipeline(
#     [
#         ('preprocessor', preprocessor),
#         ('model', vc)
#     ]
# )

# ml_pipeline_VC.fit(X_train, y_train)

# Age	Sex	ChestPainType	RestingBP	Cholesterol	FastingBS	RestingECG	MaxHR	ExerciseAngina	Oldpeak	ST_Slope
# input_data =  [[37, 'M', 'ATA', 140, 289, 0, 'Normal', 172, 'N', 1.5,	'Flat']]
# columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
# input_df = pd.DataFrame(input_data, columns=columns)


# print(input_df)
# print(type(input_data))
# print(len(input_data))
# print(type(input_df))

ml_pipeline_VC = joblib.load("aux/ml_pipeline_VC.pkl")

# preprossed_inpud_data = preprocessor.transform(input_df)
# print(preprossed_inpud_data)


# print(ml_pipeline_VC.predict(input_df))

# print(ml_pipeline_VC2.predict(input_df))

# print(ml_pipeline_VC2)

st.write(
    """
# WEB-приложение для распознавания потенциальной болезни сердца.

Заполните параметры.
"""
)

# Ввод данных с помощью виджетов Streamlit
age = st.number_input("Введите ваш возраст", min_value=0, max_value=99, value=30)
gender = st.selectbox("Выберите ваш пол (M - male, F - female)", ("M", "F"))
chest_pain_type = st.selectbox(
    "Выберите тип боли в груди (TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic)",
    ("ATA", "TA", "NAP", "ASY"),
)
resting_bp = st.number_input(
    "Введите ваше артериальное давление в покое mm Hg",
    min_value=0,
    max_value=300,
    value=120,
)
cholesterol = st.number_input(
    "Введите ваш уровень холестерина mm/dl", min_value=0, max_value=1000, value=200
)
fasting_bs = st.number_input(
    "Уровень сахара в крови натощак (1: if FastingBS > 120 mg/dl, 0: otherwise)",
    min_value=0,
    max_value=1,
    value=0,
)
resting_ecg = st.selectbox(
    "Выберите результат ЭКГ в покое (Normal, ST, LVH)", ("Normal", "ST", "LVH")
)
max_hr = st.number_input(
    "Введите ваш максимальный пульс (Numeric value between 60 and 202)",
    min_value=60,
    max_value=202,
    value=120,
)
exercise_angina = st.selectbox(
    "Есть ли у вас стенокардия при физической нагрузке? (Y: Yes, N: No)", ("N", "Y")
)
oldpeak = st.number_input(
    "Введите значение депрессии ST-сегмента (Numeric value measured in depression)",
    min_value=-5.0,
    max_value=10.0,
    step=0.1,
    value=0.0,
)
st_slope = st.selectbox(
    "Выберите наклон ST-сегмента (Up: upsloping, Flat: flat, Down: downsloping)",
    ("Up", "Down", "Flat"),
)

# Преобразование введенных данных в список
input_data = [
    age,
    gender,
    chest_pain_type,
    resting_bp,
    cholesterol,
    fasting_bs,
    resting_ecg,
    max_hr,
    exercise_angina,
    oldpeak,
    st_slope,
]

# Кнопка для добавления значений в список
if st.button("Прогнозировать наличие болезни сердца"):
    # Вывод списка в консоль
    # Age	Sex	ChestPainType	RestingBP	Cholesterol	FastingBS	RestingECG	MaxHR	ExerciseAngina	Oldpeak	ST_Slope
    data = [input_data]
    columns = [
        "Age",
        "Sex",
        "ChestPainType",
        "RestingBP",
        "Cholesterol",
        "FastingBS",
        "RestingECG",
        "MaxHR",
        "ExerciseAngina",
        "Oldpeak",
        "ST_Slope",
    ]
    input_df = pd.DataFrame(data, columns=columns)

    if ml_pipeline_VC.predict(input_df) == 0:
        st.write(
            '<span style="font-size:24px; color:green; font-weight:bold;">Поздравляем! Вероятней всего у вас нет болезни сердца! 🥳</span>',
            unsafe_allow_html=True,
        )
    else:
        st.write(
            '<span style="font-size:24px; color:red; font-weight:bold;">Скорее всего у вас проблемы с сердцем. 😬 Обратитесь к специалисту! 🤒</span>',
            unsafe_allow_html=True,
        )

    # print(ml_pipeline_VC)
    # print(ml_pipeline_VC.predict(input_df))
