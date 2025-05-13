import pandas as pd

def cargar_datos():
    datos = pd.read_csv('datos_atletas.csv')
    datos = datos.dropna()  
    datos['Atleta'] = datos['Atleta'].map({'Fondista': 1, 'Velocista': 0})
    return datos


def limpiar_outliers(df):
    columnas = ['Edad', 'Peso', 'Umbral_Lactato', 'Fibras_Lentas_%', 'Fibras_Rapidas_%']
    Q1 = df[columnas].quantile(0.25)
    Q3 = df[columnas].quantile(0.75)
    IQR = Q3 - Q1
    filtro = ~((df[columnas] < (Q1 - 1.5 * IQR)) | (df[columnas] > (Q3 + 1.5 * IQR))).any(axis=1)
    df_limpio = df[filtro]
    outliers_eliminados = (~filtro).sum()
    return df_limpio, outliers_eliminados
