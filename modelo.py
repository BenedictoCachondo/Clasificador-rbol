from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

def entrenar_modelo(df, max_depth, criterio):
    X = df[['Edad', 'Peso', 'Umbral_Lactato', 'Fibras_Lentas_%', 'Fibras_Rapidas_%']]
    y = df['Atleta']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    arbol = DecisionTreeClassifier(max_depth=max_depth, criterion=criterio)
    arbol.fit(X_train, y_train)

    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)

    return arbol, logreg, X_train, X_test, y_train, y_test

def predecir_usuario(modelo, datos_usuario):
    return modelo.predict(datos_usuario)[0]
