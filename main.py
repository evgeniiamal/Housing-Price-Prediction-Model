import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# Считывание данных
df = pd.read_csv("data/moscow_housing.csv")
print(df.shape)

X = df.drop("Price", axis=1) #признаки
y = df["Price"] #целевая переменная

# Деление данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Анализ данных
print("Размер датасета:", df.shape)
print(df.describe())

missing_values = df.isnull().sum() #подсчет пустых значений
print(missing_values)

# Построение гистограммы(визуализация данных)
plt.figure(figsize=(10, 6))
df["Price"].hist(bins=50)
plt.title("Распределение цен на квартиры")
plt.xlabel("Цена")
plt.ylabel("Количество квартир")
plt.show()

# Вычисляем корреляцию только для числовых столбцов.
correlation_matrix = df.corr(numeric_only=True)
print(correlation_matrix)
plt.figure(figsize=(10, 8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Матрица корреляции")
plt.show()

# Отделение числовых и категориальных признаков
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["str"]).columns
print("Числовые признаки:")
print(num_cols)
print("Категориальные признаки:")
print(cat_cols)

#пайплайн для числовых признаков
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())])

#пайплайн для категориальных признаков
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)])

# Создание и обучение моделей

# Линейная модель
linear_model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor",LinearRegression())])

# Дерево решений
tree_model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", DecisionTreeRegressor(random_state=42))])

# Случайный лес
forest_model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=-1))])

# Обучение моделей
models = {
    "Linear Regression": linear_model,
    "Decision Tree": tree_model,
    "Random Forest": forest_model}

results = {}

print("Обучение моделей:")

for name, model in models.items():
    print(f"\n{name}")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("\nПервые 10 предсказаний:")
    for real, pred in zip(y_test.values[:10], predictions[:10]):
        print(f"Реальная: {real:,.0f} | Предсказание: {pred:,.0f}")
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    results[name] = rmse
    print(f"RMSE: {rmse:,.2f} руб.")


# Сравнение моделей
comparison = pd.DataFrame({
    "Model": results.keys(),
    "RMSE": results.values()})

comparison = comparison.sort_values("RMSE")

print(comparison)

# Перекрестная проверка
print("CROSS VALIDATION")

scores = cross_val_score(
    forest_model,
    X_train,
    y_train,
    scoring="neg_root_mean_squared_error",
    cv=3,
    n_jobs=-1)

scores = -scores
print("Fold RMSE:")
print(scores)
print(f"\nAverage RMSE: {scores.mean():,.2f}")
print(f"Std: {scores.std():,.2f}")


# Подбор лучших гиперпараметров
print("GRID SEARCH")

param_grid = {
    "regressor__n_estimators": [100, 150],
    "regressor__max_depth": [20],
    "regressor__min_samples_split": [2]}

grid_search = GridSearchCV(
    estimator=forest_model,
    param_grid=param_grid,
    cv=3,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1)

grid_search.fit(X_train, y_train)

print("\nЛучшие параметры:")

print(grid_search.best_params_)

print("\nЛучший CV RMSE:")

print(-grid_search.best_score_)

# После подбора гиперпараметров GridSearch автоматически сохраняет модель с лучшими
# Получаем ее через best_estimator и оцениваем на ранее отложенной тестовой выборке
best_model = grid_search.best_estimator_
best_predictions = best_model.predict(X_test)
best_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        best_predictions
    )
)

print(f"Лучший тест RMSE: {best_rmse:,.2f} руб.")

# Сохранение модели
joblib.dump(best_model,"model/best_model.pkl")


