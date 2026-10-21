import pandas as pd
import streamlit as st
import plotly.express as px
from model_rb import model_rb
from quantity_per_pack import quantity_per_pack
from gas_cylinders import gas_cylinders
from autoupdating_price import autoupdating_price

st.set_page_config(
    page_title="Modelos industriales / Retail",
    page_icon=":material/analytics:",
    layout="wide",
)

PROJECTS = {
    "📿 Bandas elásticas": {
        "loader": model_rb,
        "target": "breaks",
        "description": "Clasifica si una banda elástica se romperá según sus dimensiones. Se entrena al modelo para que aprenda la resistencia y el punto de rotura de una banda elástica.",
    },
    "🥃 Cantidad por paquete": {
        "loader": quantity_per_pack,
        "target": "full",
        "description": "Clasifica si un envase está lleno a partir de sus características.",
    },
    "⛽ Cilindros de gas": {
        "loader": gas_cylinders,
        "target": "will_explode",
        "description": "Estima el riesgo de explosión a partir de las condiciones del cilindro.",
    },
    "🏪 Precio dinámico": {
        "loader": autoupdating_price,
        "target": "new_price",
        "description": "Estima el precio actualizado según el producto, el stock y las visitas. Mientras más visitas y menos stock, aumentan los precios, y mientras menos visitas y más stock, bajan los precios.",
    },
}


@st.cache_resource(show_spinner="Cargando y entrenando el modelo...")
def load_project(project_name):
    return PROJECTS[project_name]["loader"]()


def render_prediction(model, features, target):
    values = {}
    with st.form("prediction_form"):
        for column in features.columns:
            label = column.replace("_", " ").capitalize()
            series = features[column]
            if not pd.api.types.is_numeric_dtype(series):
                options = sorted(series.dropna().unique().tolist())
                values[column] = st.selectbox(label, options, key=f"input_{column}")
            else:
                minimum = float(series.min())
                maximum = float(series.max())
                default = float(series.median())
                is_integer = series.dropna().map(lambda value: float(value).is_integer()).all()
                step = 1.0 if is_integer else 0.1
                values[column] = st.number_input(
                    label,
                    min_value=minimum,
                    max_value=maximum,
                    value=default,
                    step=step,
                    key=f"input_{column}",
                )

        submitted = st.form_submit_button("Calcular predicción", type="primary")

    if submitted:
        prediction_input = pd.DataFrame([values], columns=features.columns)
        prediction = model.predict(prediction_input)[0]
        st.metric(target.replace("_", " ").capitalize(), f"{prediction:.2f}" if isinstance(prediction, float) else prediction)
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(prediction_input).max() * 100
            st.caption(f"Probabilidad de la clase seleccionada: {probability:.1f}%")

st.title("🤖 Modelos industriales / Retail")
selected_project = st.selectbox("Selecciona un modelo", list(PROJECTS))

project = PROJECTS[selected_project]
result = load_project(selected_project)
dataset = result["df"]
target = project["target"]
features = dataset.drop(columns=target)

st.title(selected_project)    
st.caption(project["description"])

data_tab, graphics, metrics_tab, prediction_tab = st.tabs(
    ["Datos", "Gráficos", "Métricas", "Predicción"]
)

with prediction_tab:
    st.subheader("Prueba el modelo")
    render_prediction(result["model"], features, target)

with metrics_tab:
    st.subheader("Rendimiento del modelo")
    st.dataframe(
        result["metrics"].astype(str),
        alt="Métricas de evaluación del modelo",
    )

with data_tab:
    st.subheader("Resumen del conjunto de datos")
    st.dataframe(
        dataset.describe(include="all").transpose(),
        alt="Estadísticas descriptivas de las variables del conjunto de datos",
    )
    st.subheader("Registros Crudos")
    st.caption(f"{len(dataset):,} filas")
    st.dataframe(dataset, hide_index=True, alt="Registros del conjunto de datos")

with graphics:
    st.subheader("Explora los datos")
    numeric_columns = dataset.select_dtypes(include="number").columns.tolist()
    category_columns = [
        column
        for column in dataset.columns
        if column == target
        or not pd.api.types.is_numeric_dtype(dataset[column])
        or dataset[column].nunique(dropna=True) <= 20
    ]
    chart_type = st.selectbox(
        "Tipo de gráfico",
        ["Distribución", "Relación entre variables", "Comparación por grupos"],
    )
    chart_data = dataset.assign(_class=dataset[target].astype(str))
    target_label = target.replace("_", " ").capitalize()

    if chart_type == "Distribución":
        column = st.selectbox("Variable", dataset.columns, key="distribution_column")
        figure = px.histogram(
            chart_data,
            x=column,
            color="_class",
            barmode="overlay",
            opacity=0.75,
            labels={
                column: column.replace("_", " ").capitalize(),
                "_class": target_label,
            },
        )
    elif chart_type == "Relación entre variables":
        x_column, y_column = st.columns(2)
        x = x_column.selectbox("Eje X", numeric_columns, key="relation_x")
        y_options = [
            column for column in numeric_columns if column != x
        ] or numeric_columns
        y = y_column.selectbox("Eje Y", y_options, key="relation_y")
        figure = px.scatter(
            chart_data,
            x=x,
            y=y,
            color="_class",
            labels={
                x: x.replace("_", " ").capitalize(),
                y: y.replace("_", " ").capitalize(),
                "_class": target_label,
            },
        )
    else:
        group_column = st.selectbox(
            "Agrupar por",
            category_columns,
            key="comparison_group",
        )
        metric_options = [
            column for column in numeric_columns if column != group_column
        ]
        metric = st.selectbox(
            "Promediar variable numérica",
            metric_options,
            key="comparison_metric",
        )
        grouped_data = (
            dataset.groupby(group_column, dropna=False, as_index=False)[metric]
            .mean()
        )
        figure = px.bar(
            grouped_data,
            x=group_column,
            y=metric,
            labels={
                group_column: group_column.replace("_", " ").capitalize(),
                metric: f"Promedio de {metric.replace('_', ' ')}",
            },
        )

    st.plotly_chart(figure, use_container_width=True)