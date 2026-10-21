import numpy as np
import pandas as pd
import plotly.express as px

def quantity_per_pack():
    np.random.seed(42)
    size = 1000

    container_type = np.random.choice(["A", "B", "C", "D", "E", "F", "G"], size)

    data = pd.DataFrame({
        "container_type": container_type
    })

    cont_esperado = data.container_type.map({"A": 120, "B": 200, "C": 400, "D": 500, "E":700, "F":900, "G":1000})
    data["container_capacity"] = cont_esperado

    # variación continua realista en el llenado de la máquina (ruido del proceso continuo)
    desviacion_llenado = np.random.normal(loc=-0.5, scale=1.5, size=size)
    data["container_content"] = np.clip(data.container_capacity + desviacion_llenado, 0, data.container_capacity)
    data["container_content"] = data["container_content"].round(1)

    # solo está lleno (1) si el contenido es exactamente igual a la capacidad
    data["full"] = np.where(data.container_content >= data.container_capacity, 1, 0)

    # ruido en el sensor real 1.5% de anomalías/fallas de lectura al azar para mantener métricas realistas
    error_sensor = np.random.rand(size) < 0.015
    data["full"] = np.where(error_sensor, 1 - data["full"], data["full"])

    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.base import BaseEstimator, TransformerMixin

    # Creamos un extractor para calcular la proporción de llenado de forma matemática en el pipeline
    class CalculadorProporcion(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self
        def transform(self, X):
            X_copy = X.copy()
            X_copy['proporcion_llenado'] = X_copy['container_content'] / X_copy['container_capacity']
            return X_copy[['proporcion_llenado']]

    X = data.drop(columns=["full"])
    y = data.full

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # Pipeline optimizado con ingeniería de características y bosque aleatorio
    model = Pipeline(steps=[
        ("calculador_ratio", CalculadorProporcion()),
        ("regressor", RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    classi = classification_report(y_test, y_pred)
    cross = cross_val_score(model, X, y, cv=5)
    cross_mean = cross.mean()

    print(f"Accuracy final (con ruido realista): {accuracy:.4f}")
    print(f"Classification Report:\n{classi}")

    metrics = pd.DataFrame({
        "Accuracy": [accuracy],
        "Classification": [str(classi)],
        "Cross": [str(cross)],
        "Cross_mean": [str(cross_mean)]
    })

    metrics = metrics.T


    return {
        "df": data,
        "model": model,
        "metrics":metrics
    }