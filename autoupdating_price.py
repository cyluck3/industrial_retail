def autoupdating_price():
    import numpy as np
    import pandas as pd
    import plotly as px

    np.random.seed(42)
    size = 1000

    product = np.random.choice(["A", "B", "C", "D", "E"], size=size)
    stock_quantity = np.random.randint(1,100, size=size)
    supplier_price = np.random.randint(1,100, size=size)
    old_price = supplier_price * 1.09
    visits_clients = np.random.randint(1,200, size=size)
    new_price = np.where(visits_clients > stock_quantity, old_price * 1.09, old_price * 0.95)

    dataset = {
        "product": product,
        "stock_quantity": stock_quantity,
        "supplier_price": supplier_price,
        "old_price": old_price,
        "visits_clients": visits_clients,
        "new_price": new_price
    }

    df = pd.DataFrame(dataset)

    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score

    X = df.drop("new_price", axis=1)
    y = df["new_price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    num_feature = ["stock_quantity", "supplier_price", "old_price", "visits_clients"]
    cat_feature = ["product"]

    num_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_transformer, num_feature),
        ("cat", cat_transformer, cat_feature)
    ])

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(random_state=42))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    cross = cross_val_score(model, X, y, cv=5)
    mean_cross = cross.mean()

    print("MSE:", mse)
    print("R2:", r2)
    print("Cross Validation:", cross)
    print("Cross Mean:", mean_cross)

    print("")
    # Automatically calculate old_price based on supplier_price
    supplier_price_for_prediction = 10 # Precio de compra
    old_price_for_prediction = supplier_price_for_prediction * 1.09 # Precio de venta

    print("New Price: ", model.predict(pd.DataFrame({
        "product": ["A"],
        "stock_quantity": [80],
        "supplier_price": [supplier_price_for_prediction],
        "old_price": [old_price_for_prediction],
        "visits_clients": [100]
    })))

    print("")

    metrics = pd.DataFrame({
        "R2-Score": r2,
        "MSE": [str(mse)],
        "Cross": [str(cross)],
        "Cross_mean": [str(mean_cross)]
    })

    metrics = metrics.T


    return {
        "df": df,
        "model": model,
        "metrics":metrics
    }
