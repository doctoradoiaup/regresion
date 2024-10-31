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

def load_data(file):
    data = pd.read_excel(file)
    return data

def preprocess_data(data):
    # Revisar datos faltantes
    st.write("Datos faltantes por columna:")
    st.write(data.isnull().sum())
    
    # Eliminar filas con datos faltantes (o puedes elegir otra estrategia)
    data = data.dropna()
    
    # Codificar variable objetivo
    y = data["Renuncia"].map({"Sí": 1, "No": 0})
    X = data.drop(columns=["Renuncia"])
    
    # Convertir variables categóricas en variables dummy
    X = pd.get_dummies(X, drop_first=True)
    
    return X, y

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
            "y_pred_proba": y_pred_proba,
            "X_test": X_test,
        }
        
    return results

def display_model_summary(results):
    summary_data = []

    for name, result in results.items():
        st.subheader(f"Resumen del modelo: {name}")
        st.write(f"Precisión: {result['accuracy']:.2f}")
        st.write("Matriz de confusión:")
        cm = result['confusion_matrix']
        st.write(cm)

        report = result['report']
        
        for key in ["0", "1", "accuracy", "macro avg", "weighted avg"]:
            if key in report:
                summary_data.append({
                    "Modelo": name,
                    "Tipo": key,
                    "Precisión": report[key].get('precision', np.nan),
                    "Recall": report[key].get('recall', np.nan),
                    "F1-Score": report[key].get('f1-score', np.nan),
                    "Soporte": report[key].get('support', np.nan)
                })
        
        prob_df = pd.DataFrame({
            'ID': result['X_test'].index,
            'Probabilidad de Renuncia (1)': result['y_pred_proba']
        })
        
        st.write(f"Probabilidades de Renuncia (1) para cada instancia - {name}:")
        st.dataframe(prob_df)

    summary_df = pd.DataFrame(summary_data)
    st.write("Resumen de métricas de evaluación:")
    st.dataframe(summary_df)

def main():
    st.title("Modelo de Predicción de Renuncia")
    
    uploaded_file = st.file_uploader("Cargar archivo Excel", type=["xlsx"])
    if uploaded_file:
        data = load_data(uploaded_file)

        st.subheader("Datos Cargados")
        st.dataframe(data)

        # Preprocesar datos
        X, y = preprocess_data(data)

        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Entrenar y evaluar modelos
        results = train_and_evaluate(X_train, X_test, y_train, y_test)

        # Mostrar resumen del modelo
        display_model_summary(results)

if __name__ == "__main__":
    main()
