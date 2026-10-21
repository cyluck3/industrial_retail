import numpy as np
import pandas as pd

def model_rb(): # Resistence Bands
    np.random.seed(42)
    size = 1000


    # Se entrena al modelo para que aprenda la resistencia y el punto de rotura de una banda elástica

    unstretched_length = np.random.randint(30, 90, size)
    max_long_capability = unstretched_length + np.random.randint(10,20, size)
    stretched_length = unstretched_length + np.random.randint(10, 20, size) * np.random.normal(1,0.09, size)
    breaks = np.where(stretched_length > max_long_capability, 1, 0)

    data = pd.DataFrame({
        'unstretched_length': unstretched_length,
        'max_long_capability': max_long_capability,
        'stretched_length': stretched_length,
        'breaks': breaks
    })


    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    X = data.drop('breaks', axis=1)
    y = data['breaks']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    num_features = ['unstretched_length', 'max_long_capability', 'stretched_length']

    num_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_features)
    ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression())
    ])

    model.fit(X_train, y_train)

    y_predict = model.predict(X_test)

    print(model.score(X_test, y_test))

    from sklearn.metrics import confusion_matrix, classification_report

    confussion_matrix = confusion_matrix(y_test, y_predict)
    tn, fp, fn, tp = confussion_matrix.ravel()

    classification = classification_report(y_test, y_predict)
    cross = cross_val_score(model, X, y, cv=5)
    cross_mean = cross.mean()

    metrics = pd.DataFrame({
        "True Positives": [tp],
        "False Positives": [fp],
        "True Negatives": [tn],
        "False Negatives": [fn],
        "Classification": str(classification),
        "Cross": str(cross),
        "Cross_mean": str(cross_mean)
    })

    metrics = metrics.T


    return {
        "df": data,
        "model": model,
        "metrics":metrics
    }
