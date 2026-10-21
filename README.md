# Industrial and Retail Models

A Streamlit app for exploring small machine-learning models through interactive data, charts, evaluation metrics, and predictions.

## Models

- **Elastic bands:** predicts whether a band will break based on its dimensions.
- **Package fill:** predicts whether a container is full.
- **Gas cylinders:** estimates explosion risk from cylinder conditions.
- **Dynamic pricing:** estimates an updated price based on product, stock, and customer visits.

The app generates example datasets and trains each model when selected. The trained model is cached for the session. The Charts tab provides distribution, variable relationship, and group comparison views.

## Run locally

Install the dependencies:

```bash
pip install streamlit pandas numpy scikit-learn plotly
```

Start the app from the project directory:

```bash
streamlit run app.py
```
