import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.tree import plot_tree

from preprocesamiento import cargar_datos, limpiar_outliers
from modelo import entrenar_modelo, predecir_usuario

st.set_page_config(page_title='Clasificador de Atletas', layout='wide')

# Sidebar
def sidebar_parametros(df):
    st.sidebar.header("Parámetros del Clasificador")
    max_depth = st.sidebar.slider("Profundidad del árbol", 2, 4, 3)
    criterio = st.sidebar.selectbox("Criterio", ['gini', 'entropy'])

    st.sidebar.header("Datos del Atleta")
    edad = st.sidebar.slider("Edad", int(df['Edad'].min()), int(df['Edad'].max()), int(df['Edad'].mean()))
    peso = st.sidebar.slider("Peso", int(df['Peso'].min()), int(df['Peso'].max()), int(df['Peso'].mean()))
    umbral = st.sidebar.slider("Umbral Lactato", float(df['Umbral_Lactato'].min()), float(df['Umbral_Lactato'].max()), float(df['Umbral_Lactato'].mean()))
    fibras_lentas = st.sidebar.slider("Fibras Lentas %", 0.0, 100.0, float(df['Fibras_Lentas_%'].mean()))
    fibras_rapidas = st.sidebar.slider("Fibras Rápidas %", 0.0, 100.0, float(df['Fibras_Rapidas_%'].mean()))

    return max_depth, criterio, edad, peso, umbral, fibras_lentas, fibras_rapidas

# Main App
def main():
    datos = cargar_datos()
    datos_limpios, outliers_eliminados = limpiar_outliers(datos)

    max_depth, criterio, edad, peso, umbral, fibras_lentas, fibras_rapidas = sidebar_parametros(datos_limpios)

    arbol, logreg, X_train, X_test, y_train, y_test = entrenar_modelo(datos_limpios, max_depth, criterio)

    pred_arbol = arbol.predict(X_test)
    acc_arbol = accuracy_score(y_test, pred_arbol)

    pagina = st.sidebar.radio("Páginas", ["Métricas y Predicción", "Gráficos"])

    if pagina == "Métricas y Predicción":
        st.title("Evaluación y Predicción del Modelo")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Métricas del Árbol")
            st.write(f"Precisión: {acc_arbol:.2f}")
            st.text(classification_report(y_test, pred_arbol))
            st.write(f"Outliers eliminados: {outliers_eliminados}")

        with col2:
            st.subheader("Predicción del Usuario")
            datos_usuario = pd.DataFrame([[edad, peso, umbral, fibras_lentas, fibras_rapidas]],
                                         columns=['Edad', 'Peso', 'Umbral_Lactato', 'Fibras_Lentas_%', 'Fibras_Rapidas_%'])
            pred = predecir_usuario(arbol, datos_usuario)
            resultado = "Fondista" if pred == 1 else "Velocista"
            st.success(f"El modelo predice que el atleta es: **{resultado}**")

    else:
        st.title("Visualización del Modelo")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Matriz de Confusión")
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, pred_arbol), annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)

            st.subheader("Curva ROC")
            proba_arbol = arbol.predict_proba(X_test)[:, 1]
            proba_log = logreg.predict_proba(X_test)[:, 1]
            fpr_arbol, tpr_arbol, _ = roc_curve(y_test, proba_arbol)
            fpr_log, tpr_log, _ = roc_curve(y_test, proba_log)
            auc_arbol = auc(fpr_arbol, tpr_arbol)
            auc_log = auc(fpr_log, tpr_log)

            fig, ax = plt.subplots()
            ax.plot(fpr_arbol, tpr_arbol, label=f"Árbol (AUC={auc_arbol:.2f})", color='blue')
            ax.plot(fpr_log, tpr_log, label=f"Regresión Logística (AUC={auc_log:.2f})", color='green')
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel("Falsos Positivos")
            ax.set_ylabel("Verdaderos Positivos")
            ax.set_title("Curva ROC")
            ax.legend()
            st.pyplot(fig)

            st.subheader("Árbol de Decisión")
            fig, ax = plt.subplots(figsize=(6, 4))
            plot_tree(arbol, filled=True, feature_names=X_train.columns, class_names=["Velocista", "Fondista"], ax=ax)
            st.pyplot(fig)

        with col2:
            st.subheader("Distribución de Variables")
            fig, axs = plt.subplots(5, 1, figsize=(6, 12))

            sns.histplot(datos_limpios['Edad'], kde=True, ax=axs[0], color='skyblue')
            axs[0].set_title("Edad")

            sns.histplot(datos_limpios['Peso'], kde=True, ax=axs[1], color='lightgreen')
            axs[1].set_title("Peso")

            sns.histplot(datos_limpios['Umbral_Lactato'], kde=True, ax=axs[2], color='salmon')
            axs[2].set_title("Umbral de Lactato")

            sns.histplot(datos_limpios['Fibras_Lentas_%'], kde=True, ax=axs[3], color='lightgreen')
            axs[3].set_title("Fibras Lentas %")

            sns.histplot(datos_limpios['Fibras_Rapidas_%'], kde=True, ax=axs[4], color='lightcoral')
            axs[4].set_title("Fibras Rápidas %")

            plt.tight_layout()
            st.pyplot(fig)

if __name__ == "__main__":
    main()
