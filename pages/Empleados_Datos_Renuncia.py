# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 12:57:22 2024

@author: jperezr
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


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

# Función para cargar el archivo y procesar los datos
def load_data():
    uploaded_file = st.file_uploader("Cargar archivo Excel", type=["xlsx"])
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        st.write("Datos cargados:")
        st.dataframe(df)
        
        # Convertir columnas categóricas a variables dummy
        df_dummies = pd.get_dummies(df, drop_first=True)
        st.write("Datos con variables dummy:")
        st.dataframe(df_dummies)
        
        return df, df_dummies
    return None, None

# Función para entrenar y evaluar modelos
def train_and_evaluate(X_train, X_test, y_train, y_test):
    models = {
        "Regresión Logística": LogisticRegression(max_iter=1000),
        "Árbol de Decisión": DecisionTreeClassifier(),
        "Bosque Aleatorio": RandomForestClassifier()
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        results[name] = {
            "accuracy": accuracy,
            "confusion_matrix": cm,
            "report": report,
            "model": model
        }
        
    return results

# Función para mostrar el resumen del modelo
def display_model_summary(results, X):
    summary_data = []

    for name, result in results.items():
        st.subheader(f"Resumen del modelo: {name}")
        st.write(f"Precisión: {result['accuracy']:.2f}")
        st.write("Matriz de confusión:")
        cm = result['confusion_matrix']
        st.write(cm)
        
        # Mostrar la interpretación de la matriz de confusión
        interpret_confusion_matrix(cm)

        # Mostrar el reporte de clasificación
        report = result['report']
        
        # Añadir métricas al DataFrame
        for key in ["0", "1", "accuracy", "macro avg", "weighted avg"]:
            if key in report:
                summary_data.append({
                    "Modelo": name,
                    "Tipo": key,
                    "Precisión": report[key].get('precision', np.nan) if isinstance(report[key], dict) else np.nan,
                    "Recall": report[key].get('recall', np.nan) if isinstance(report[key], dict) else np.nan,
                    "F1-Score": report[key].get('f1-score', np.nan) if isinstance(report[key], dict) else np.nan,
                    "Soporte": report[key].get('support', np.nan) if isinstance(report[key], dict) else np.nan
                })

        # Gráficos
        if name == "Regresión Logística":
            # Curva ROC
            fpr, tpr, _ = roc_curve(y_test, result['model'].predict_proba(X_test)[:, 1])
            fig, ax = plt.subplots()  # Crear figura y ejes
            ax.plot(fpr, tpr, label='Curva ROC')
            ax.plot([0, 1], [0, 1], linestyle='--')
            ax.set_title("Curva ROC - Regresión Logística")
            ax.set_xlabel("Tasa de Falsos Positivos")
            ax.set_ylabel("Tasa de Verdaderos Positivos")
            st.pyplot(fig)  # Pasar la figura a st.pyplot()

            # Importancia de las características
            log_reg_coef = pd.Series(result['model'].coef_[0], index=X.columns)
            log_reg_coef = log_reg_coef.sort_values(ascending=False)

            # Mostrar coeficientes de la regresión logística
            coef_df = pd.DataFrame({
                'Característica': log_reg_coef.index,
                'Coeficiente': log_reg_coef.values,
                'Interpretación': np.where(log_reg_coef > 0, 
                                            "Aumenta probabilidad de renuncia", 
                                            "Disminuye probabilidad de renuncia")
            })
            st.write("Coeficientes de la Regresión Logística:")
            st.dataframe(coef_df)

            fig, ax = plt.subplots(figsize=(10, 10))  # Crear figura y ejes
            sns.barplot(x=log_reg_coef.values, y=log_reg_coef.index, palette='viridis', ax=ax, width=0.1)
            ax.set_title('Importancia de las características - Regresión Logística')
            ax.set_xlabel('Coeficientes')
            ax.set_ylabel('Características')
            ax.set_xlim(-1, 1)
            ax.grid(axis='x')
            plt.tight_layout()
            st.pyplot(fig)  # Pasar la figura a st.pyplot()

        elif name in ["Árbol de Decisión", "Bosque Aleatorio"]:
            # Importancia de las características
            importance = result['model'].feature_importances_
            importance_df = pd.DataFrame(importance, index=X.columns, columns=['Importancia']).sort_values(by='Importancia', ascending=False)
            
            importance_df['Interpretación'] = np.where(importance_df['Importancia'] > 0, 
                                                        "Contribuye a la predicción", 
                                                        "No contribuye a la predicción")
            st.write(f"Importancia de las características - {name}:")
            st.dataframe(importance_df)
            
            fig, ax = plt.subplots(figsize=(8, len(importance_df) * 0.2))  # Crear figura y ejes
            sns.barplot(x=importance_df.Importancia, y=importance_df.index, palette="viridis", ax=ax, width=0.4)
            ax.set_title(f"Importancia de las características - {name}")
            ax.set_xlabel("Importancia")
            ax.set_ylabel("Características")
            ax.grid(axis='x')
            plt.tight_layout()
            st.pyplot(fig)  # Pasar la figura a st.pyplot()

    summary_df = pd.DataFrame(summary_data)
    st.write("Resumen de métricas de evaluación:")
    st.dataframe(summary_df)

# Función para interpretar la matriz de confusión
def interpret_confusion_matrix(cm):
    VN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    VP = cm[1, 1]

    st.write("Predicción 0 (No Renuncia):")
    st.write(f"{VN}: Verdaderos negativos (VN): {VN} casos donde el modelo predijo 'no renuncia' y efectivamente no renunciaron.")
    st.write(f"{FP}: Falsos positivos (FP): {FP} casos donde el modelo predijo 'no renuncia' pero en realidad renunciaron.")
    
    st.write("Predicción 1 (Renuncia):")
    st.write(f"{FN}: Falsos negativos (FN): {FN} casos donde el modelo predijo 'renuncia' pero en realidad no renunciaron.")
    st.write(f"{VP}: Verdaderos positivos (VP): {VP} casos donde el modelo predijo 'renuncia' y efectivamente renunciaron.")

# Main
st.title("Análisis de Renuncia de Empleados")
df, df_dummies = load_data()

if df is not None:
    # Suponiendo que 'Renuncia' es la columna objetivo
    X = df_dummies.drop(columns=['Renuncia_Si'])  # Asegúrate de que esta es la codificación correcta
    y = df_dummies['Renuncia_Si']

    # Dividir los datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar y evaluar modelos
    results = train_and_evaluate(X_train, X_test, y_train, y_test)

    # Mostrar resúmenes de los modelos
    display_model_summary(results, X)
