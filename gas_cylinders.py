def gas_cylinders():
    import numpy as np
    import pandas as pd
    import plotly.express as px

    np.random.seed(42)
    size = 1000

    cilindro_capacidad = np.random.choice([10, 18, 43], size)
    # Modifico el llenado para que algunos cilindros puedan estar sobrellenados (por encima del 85% e incluso del 100% por fallos de válvula)
    fill_percentage_base = np.random.uniform(70, 95, size)
    llenado = cilindro_capacidad * (fill_percentage_base / 100)
    temperatura = np.random.randint(15, 55, size) # Ampliamos rango de temperatura (climas extremos o exposición al sol)

    antiguedad = np.random.randint(0, 20, size) # Antigüedad en años

    # Presión aproximada, aumenta con la temperatura y fuertemente si el espacio vacío es muy pequeño (sobrellenado)
    espacio_vacio_porcentaje = 100 - fill_percentage_base
    presion = (temperatura * 0.5) + (100 / (espacio_vacio_porcentaje + 1)) * 1.5
    # Añadir un factor de ruido aleatorio a la presión
    presion += np.random.normal(0, 2, size)

    # Resistencia máxima del cilindro: decae con los años debido a corrosión y fatiga
    resistencia_base = 45 # Presión nominal máxima soportada en bar
    resistencia_maxima = resistencia_base - (antiguedad * 0.8) - np.random.uniform(0, 5, size)

    # 3. Variable Objetivo: ¿Explotará?
    will_explode = (presion > resistencia_maxima).astype(int)

    # Crear el DataFrame enriquecido
    df= pd.DataFrame({
        'capacity_kg': cilindro_capacidad,
        'fill_weight_kg': np.round(llenado, 2),
        'fill_percentage': np.round(fill_percentage_base, 2),
        'temperature_c': temperatura,
        'age_years': antiguedad,
        'pressure_bar': np.round(presion, 2),
        'max_pressure_bar': np.round(resistencia_maxima, 2),
        'will_explode': will_explode
    })

    # Mostrar balance de la variable objetivo y las primeras filas
    print(f"Distribución de Explosiones:\n{df['will_explode'].value_counts(normalize=True) * 100}\n")

    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier

    X = df.drop("will_explode", axis=1)
    y = df["will_explode"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    num_features = ["capacity_kg", "fill_weight_kg", "fill_percentage", "temperature_c", "age_years", "pressure_bar", "max_pressure_bar"]

    num_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_transformer, num_features)
    ])

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    model.fit(X_train, y_train)

    y_predict = model.predict(X_test)

    from sklearn.metrics import confusion_matrix, r2_score

    r2 = r2_score(y_test, y_predict)
    confm = confusion_matrix(y_test, y_predict)
    cross = cross_val_score(model, X, y, cv=5)
    cross_mean = cross.mean()

    print(f"R2 Score: {r2}")
    print("Confusion Matrix:\n", confm)
    print(f"Cross Validation Scores: {cross}")
    print(f"Cross Mean: {cross_mean}")

    print("")

    cross = cross_val_score(model, X, y, cv=5)
    cross_mean = cross.mean()

    metrics = pd.DataFrame({
        "R2-Score": r2,
        "Confussion Matrix": [str(confm)],
        "Cross": [str(cross)],
        "Cross_mean": [str(cross_mean)]
    })

    metrics = metrics.T


    return {
        "df": df,
        "model": model,
        "metrics":metrics
    }
