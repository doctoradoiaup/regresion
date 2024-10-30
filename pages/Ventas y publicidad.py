# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 12:58:51 2024

@author: jperezr
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

# Estilo de fondo
page_bg_img = """
<style>
[data-testid="stAppViewContainer"]{
background:
radial-gradient(black 15%, transparent 16%) 0 0,
radial-gradient(black 15%, transparent 16%) 8px 8px,
radial-gradient(rgba(255,255,255,.1) 15%, transparent 20%) 0 1px,
radial-gradient(rgba(255,255,255,.1) 15%, transparent 20%) 8px 9px;
background-color:#282828;
background-size:16px 16px;
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

st.title("Análisis de Ventas y Publicidad con Modelo de Regresión Lineal Múltiple")
st.write("""
Sube tus datos de ventas y publicidad en un archivo Excel y selecciona columnas relacionadas con publicidad.
""")

# Leer el archivo Excel
@st.cache_data
def load_data(file):
    return pd.read_excel(file)

# Cargar el archivo Excel
file = st.file_uploader("Sube tu archivo Excel", type=["xlsx"])
if file:
    data = load_data(file)

    # Mostrar el DataFrame
    st.write(data)

    # Seleccionar columnas
    options = ["TV", "Radio", "Periodicos"]
    selected_columns = st.multiselect("Selecciona columnas", options)

    if selected_columns:
        # Configurar variables dependientes e independientes
        X = data[selected_columns]
        y = data["Ventas"]

        # Seleccionar el modelo a usar
        model_choice = st.selectbox("Selecciona el modelo de regresión", 
                                    ["Regresión Lineal", "Ridge Regression", "Lasso Regression", "Elastic Net",
                                     "SVR", "Random Forest", "KNN", "Polynomial Regression", 
                                     "Gradient Boosting", "Decision Tree", "Bayesian Ridge", 
                                     "Huber Regression", "Theil-Sen"])

        # Inicializar el modelo basado en la selección
        if model_choice == "Regresión Lineal":
            model = LinearRegression()
        elif model_choice == "Ridge Regression":
            alpha_value = st.slider("Selecciona el valor de Alpha para Ridge", 0.0, 10.0, 1.0)
            model = Ridge(alpha=alpha_value)
        elif model_choice == "Lasso Regression":
            alpha_value = st.slider("Selecciona el valor de Alpha para Lasso", 0.0, 10.0, 1.0)
            model = Lasso(alpha=alpha_value)
        elif model_choice == "Elastic Net":
            alpha_value = st.slider("Selecciona el valor de Alpha para Elastic Net", 0.0, 10.0, 1.0)
            l1_ratio = st.slider("Selecciona el ratio L1 para Elastic Net", 0.0, 1.0, 0.5)
            model = ElasticNet(alpha=alpha_value, l1_ratio=l1_ratio)
        elif model_choice == "SVR":
            model = SVR(kernel='rbf')  # Puedes cambiar el kernel si lo deseas
        elif model_choice == "Random Forest":
            n_estimators = st.slider("Número de árboles en el bosque", 1, 100, 10)
            model = RandomForestRegressor(n_estimators=n_estimators)
        elif model_choice == "KNN":
            n_neighbors = st.slider("Número de vecinos más cercanos (k)", 1, 20, 5)
            model = KNeighborsRegressor(n_neighbors=n_neighbors)
        elif model_choice == "Polynomial Regression":
            degree = st.slider("Grado del polinomio", 2, 5, 2)
            poly = PolynomialFeatures(degree=degree)
            X = poly.fit_transform(X)
            model = LinearRegression()
        elif model_choice == "Gradient Boosting":
            n_estimators = st.slider("Número de estimadores", 1, 100, 10)
            model = GradientBoostingRegressor(n_estimators=n_estimators)
        elif model_choice == "Decision Tree":
            max_depth = st.slider("Profundidad máxima del árbol", 1, 20, 5)
            model = DecisionTreeRegressor(max_depth=max_depth)
        elif model_choice == "Bayesian Ridge":
            model = BayesianRidge()
        elif model_choice == "Huber Regression":
            model = HuberRegressor()
        elif model_choice == "Theil-Sen":
            model = sm.RLM(y, sm.add_constant(X))  # Implementación robusta de Theil-Sen

        # Ajustar el modelo y predecir
        if model_choice != "Theil-Sen":
            model.fit(X, y)
            y_pred = model.predict(X)
        else:
            model = model.fit()
            y_pred = model.predict(sm.add_constant(X))

        # Evaluar el modelo
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        # Mostrar resultados
        st.subheader(f"Resultados del Modelo: {model_choice}")
        st.write(f"Error Cuadrático Medio: {mse}")
        st.write(f"R^2: {r2}")

        # Mostrar la ecuación de ajuste o resumen de parámetros según el modelo
        if model_choice in ["Regresión Lineal", "Ridge Regression", "Lasso Regression", "Elastic Net", "Bayesian Ridge", "Huber Regression"]:
            st.subheader("Ecuación de Ajuste")
            equation = "y = {:.4f}".format(model.intercept_)  # Intercepto
            equation += " + " + " + ".join("{:.4f} * {}".format(coef, col) for coef, col in zip(model.coef_, selected_columns))
            st.write(equation)

        elif model_choice == "Polynomial Regression":
            st.subheader("Ecuación de Ajuste Polinomial")
            equation = "y = {:.4f}".format(model.intercept_)
            terms = ["{:.4f} * {}".format(coef, f"x{i}") for i, coef in enumerate(model.coef_, start=1)]
            equation += " + " + " + ".join(terms)
            st.write(equation)

        elif model_choice in ["Random Forest", "Gradient Boosting", "Decision Tree"]:
            st.subheader("Importancia de las Variables")
            importance = model.feature_importances_
            importance_df = pd.DataFrame({"Variable": selected_columns, "Importancia": importance})
            st.write(importance_df)

        # Gráfico de dispersión
        plt.figure(figsize=(10, 6))
        plt.scatter(y, y_pred)
        plt.xlabel("Ventas reales")
        plt.ylabel("Ventas predichas")
        plt.title(f"Gráfico de dispersión: Ventas reales vs Ventas predichas ({model_choice})")
        plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linewidth=2)  # Línea de 45 grados
        st.pyplot(plt)

# Sección de ayuda en la barra lateral
st.sidebar.markdown(
    """
    ### Ayuda
    Esta aplicación permite realizar análisis de ventas y publicidad utilizando varios modelos de regresión.

    **Instrucciones:**
    1. **Carga tu archivo Excel** con las columnas relacionadas con ventas y publicidad.
    2. **Selecciona las columnas** para las variables independientes y el modelo de regresión.
    3. **Ajusta el modelo** y revisa los resultados, incluyendo el error cuadrático medio y el R^2.
    4. **Consulta el gráfico** para visualizar el rendimiento del modelo.

    **Requisitos del archivo:**
    - El archivo debe contener al menos las columnas 'Ventas', 'TV', 'Radio', y 'Periodicos'.
    
    ---
    **Desarrollado por: Javier Horacio Pérez Ricárdez**
    """,
    unsafe_allow_html=True
)
