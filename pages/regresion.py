# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 11:27:44 2024

@author: jperezr
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Cargar el archivo Excel
def load_data(file):
    data = pd.read_excel(file)
    return data

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
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilidades de renuncia (1)
        
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        results[name] = {
            "accuracy": accuracy,
            "confusion_matrix": cm,
            "report": report,
            "model": model,
            "y_pred_proba": y_pred_proba,  # Agregar probabilidades
            "X_test": X_test,  # Agregar X_test para crear el DataFrame después
        }
        
    return results

# Modificación en la función para mostrar el resumen del modelo
def display_model_summary(results):
    summary_data = []

    for name, result in results.items():
        st.subheader(f"Resumen del modelo: {name}")
        st.write(f"Precisión: {result['accuracy']:.2f}")
        st.write("Matriz de confusión:")
        cm = result['confusion_matrix']
        st.write(cm)

        # Mostrar el reporte de clasificación
        report = result['report']
        
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
        
        # Crear DataFrame de probabilidades de renuncia
        prob_df = pd.DataFrame({
            'ID': result['X_test'].index,  # Asegúrate de que el índice esté presente
            'Probabilidad de Renuncia (1)': result['y_pred_proba']
        })
        
        st.write(f"Probabilidades de Renuncia (1) para cada instancia - {name}:")
        st.dataframe(prob_df)

    summary_df = pd.DataFrame(summary_data)
    st.write("Resumen de métricas de evaluación:")
    st.dataframe(summary_df)

# Interfaz de Streamlit
def main():
    st.title("Modelo de Predicción de Renuncia")
    
    # Cargar el archivo
    uploaded_file = st.file_uploader("Cargar archivo Excel", type=["xlsx"])
    if uploaded_file:
        data = load_data(uploaded_file)

        # Mostrar datos
        st.subheader("Datos Cargados")
        st.dataframe(data)

        # Seleccionar características y la variable objetivo
        X = data.drop(columns=["Renuncia"])  # Reemplaza "Renuncia" con el nombre de tu columna
        y = data["Renuncia"]  # Asegúrate de que esta columna tenga valores 'Sí' y 'No'

        # Codificar variable objetivo si es necesario
        y = y.map({"Sí": 1, "No": 0})

        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Entrenar y evaluar modelos
        results = train_and_evaluate(X_train, X_test, y_train, y_test)

        # Mostrar resumen del modelo
        display_model_summary(results)

if __name__ == "__main__":
    main()
