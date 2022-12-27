#!/usr/bin/env python
# coding: utf-8

# # Probando modelos de clasificación binaria 

# En esta práctica vamos a construir modelos de clasificación para el dataset ya analizado en la práctica {ref}`exploratory`
# 
# El objetivo es clasificar si el cliente se subscribirá a un depósito a largo plazo (clasificación binaria). Recordemos las descripciones de lo que represeta cada columna:
# - **Datos del cliente**
#     - `age`: edad del candidato
#     - `job`: tipo de trabajo el candidato
#     - `marital`: estado civil
#     - `education`: nivel educativo
#     - `default`: ¿Tiene incumplimientos en algún crédito?
#     - `housing`: ¿Tiene una hipoteca?
#     - `loan`: ¿Tiene un crédito personal?
# - **Datos relacionados con el último contacto de la campaña**
#     - `contact`: canal de comunicicación de los contactos con el cliente
#     - `month`: mes del último contacto en el presente año
#     - `day_of_week`: día de la semana del último contacto en el presente año
#     - `duration`: duración de la última comunicación en segundos
# - **Historial de contactos**
#     - `campaign`: número de contactos realizados durante la campaña con el cliente
#     - `pdays`: número de días que pasaron desde que el cliente fue contactado en una anterior campaña. Si no hubo contacto, vale `999`
#     - `previous`: número de contactos realizados con el cliente antes de esta campaña
#     - `poutcome`: resultado de la anterior campaña
# - **Variables socioeconómicas**
#     - `emp.var.rate`: tasa de variación de empleo (indicador cuatrimestral)
#     - `cons.price.idx`: índice de precios de consumo (indicador mensual)
#     - `cons.conf.idx`: índice de confianza del consumidor (indicador mensual)
#     - `euribor3m`: tasa euribor a 3 meses (indicador diario)
#     - `nr.employed`: número de ocupados (indicador cuatrimestral, en miles)
# - **Variable objetivo**
#     - `y`: ¿Ha contratado el cliente un depósito a largo plazo? 

# In[16]:


import os
import urllib
import zipfile
from typing import List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import scipy.stats as ss


# In[17]:


working_dir = "."
download_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
bank_path = "data/bank"
raw_data_rel_path = "data/data_raw"
zip_path = "bank-additional/bank-additional-full.csv"
raw_data_path = os.path.join(working_dir, raw_data_rel_path)


# In[18]:


def fetch_data(
    download_url: str,
    raw_data_path: str,
    bank_path: str,
    zip_path: str
) -> None:
    os.makedirs(raw_data_path, exist_ok=True)
    zip_file_path = os.path.join(raw_data_path, "data.zip")
    urllib.request.urlretrieve(download_url, zip_file_path)
    with zipfile.ZipFile(zip_file_path) as zip_ref:
        zip_ref.extract(
            zip_path,
            bank_path
        )

def load_data(bank_path: str, zip_path: str) -> pd.DataFrame:
    csv_path = os.path.join(bank_path, zip_path)
    df = pd.read_csv(csv_path, sep=";")
    return df


# In[19]:


# funciones auxiliares

def filter_cols(df: pd.DataFrame, rem_cols: List[str]):
    cols = [col for col in df.columns if col not in rem_cols]
    df = df[cols]
    return df 


# Cargamos los datos y guardamos la variable objetivo en una serie separada

# In[20]:


fetch_data(download_url, raw_data_path, bank_path, zip_path)
bank = load_data(bank_path, zip_path)
y = bank["y"] == "yes"
bank.drop(columns="y", inplace=True)


# ## Train - Test 

# Dividimos nuestro conjunto de datos en dos conjuntos, train y test, usando una división *estratificada* para asegurarnos que la variable objetivo está igualmente representada en ambos conjuntos (recordemos que el conjunto de datos está desbalanceado)

# In[21]:


from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in sss.split(bank, y):
    bank_train, bank_test = bank.loc[train_idx], bank.loc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]


# Por ahora dejamos `bank_test` a un lado para centranos en el procesamiento de variables e implementación de modelos con el conjunto de train.

# ## Preprocesado 
# 
# Las tareas de preprocesado que vamos a llevar a cabo son 
# 
# - Vamos a eliminar la columna `duration` porque no es una variable de la que dispongamos cuando el modelo esté en producción
# - Basándonos en el análisis exploratorio de datos, nos vamos a quedar solamente con la variable `poutcome` de entre las variables que referencian campañas anteriores. 
# - Igualmente, dada la alta correlación entre alguna de las variables categóricas, nos quedamos con `no.employed` en detrimento de `euribor3m` y `emp.var.rate`.
# 
# Sería interesante realizar una **reducción de la dimensionalidad** una vez estas variables hayan sido codificadas, pero por simplificar vamos a quedarnos con las que a priori puedan funcionar mejor. 

# In[22]:


def preprocess(df: pd.DataFrame):
    rem_cols=["duration", "campaign", "pdays", "previous", "euribor3m", "emp.var.rate"]
    if len(set(rem_cols) - set(df.columns)):
        raise KeyError("Intentando borrar columnas inexistentes")
    df = filter_cols(df, rem_cols=rem_cols)
    return df


# In[23]:


bank_train_preprocessed = preprocess(bank_train)


# Separamos variables numéricas y categóricas ya que tendrán procesados diferentes

# In[24]:


num_cols = bank_train_preprocessed.select_dtypes("number").columns.to_list()
cat_cols = [col for col in bank_train_preprocessed.columns if col not in num_cols]


# In[25]:


for col in cat_cols:
    print(f"Columna {col} tiene {bank_train_preprocessed[col].nunique()} valores únicos")


# In[26]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

col_transformer = ColumnTransformer([
    ("num", StandardScaler(), num_cols), 
    ("cat", OneHotEncoder(sparse=False), cat_cols)
])
col_transformer.fit(bank_train_preprocessed)


# In[27]:


def process(df: pd.DataFrame):
    X = col_transformer.transform(df)
    one_hot_attr = col_transformer.transformers_[1][1].get_feature_names_out()
    df_processed = pd.DataFrame(X, columns=[*num_cols, *one_hot_attr])
    return df_processed


# In[28]:


bank_train_processed = process(bank_train_preprocessed)


# In[29]:


bank_train_processed


# ## Modelos

# Vamos a probar diferentes tipologías de modelos y estudiar si debemos realizar selección de variables

# In[30]:


X_train = bank_train_processed


# In[31]:


from sklearn.metrics import make_scorer, precision_score, recall_score, balanced_accuracy_score

scorers = {
    'precision_score': make_scorer(precision_score),
    'recall_score': make_scorer(recall_score),
    'balanced_accuracy': make_scorer(balanced_accuracy_score)
}


# ### Random Forest
# 
# Vamos a entrenar un Random Forest como primera aproximación y para valorar algunas métricas

# In[43]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

rf_clf = RandomForestClassifier(criterion="entropy")
rf_clf.fit(bank_train_processed, y_train)
rf_clf_scores = cross_validate(rf_clf, bank_train_processed, y_train, cv=3, scoring=scorers)
rf_clf_scores


# ### Selección de Variables en RandomForest
# 
# Vamos a seleccionar algunas variables utilizando el atributo `feature_importances_` de estimador Random Forest

# In[47]:


from sklearn.feature_selection import SelectFromModel

def feature_selection_from_model(estimator, X_df, y, ax=None):
    # Ajustamos SelectFromModel (basado en improtancia de pesos)
    selector = SelectFromModel(estimator)
    selector.fit(X_df, y)
    estimator = selector.estimator_
    # Características seleccionadas
    model_selection_features = selector.get_feature_names_out()
    # Obtenemos importancias de las características
    n_features = X_df.shape[1]
    importances = estimator.feature_importances_
    # Si las importancias vienen como array multidimensional hacemos la media
    if importances.shape != (n_features,):
        importances = estimator.feature_importances_.mean(axis=0)
            
    print('Número de características seleccionadas :', len(model_selection_features),'\n')
    print('Características seleccionadas :', model_selection_features,'\n')
    
    if ax is not None:
        #Realizamos un grafico de barras con la importancia de las características
        ax.barh(range(n_features), importances, align='center') 
        ax.set_yticks(np.arange(n_features)) 
        ax.set_yticklabels(X_df.columns.values)
        ax.set_xlabel('Feature importance')
        ax.set_ylabel('Feature')
    
    return set(model_selection_features)


# In[48]:


fig, ax = plt.subplots(figsize=(8, 8))

rf_selected_features = feature_selection_from_model(
    RandomForestClassifier(), 
    X_train, 
    y_train, 
    ax=ax
)


# In[49]:


X_train_selected = X_train[list(rf_selected_features)]


# ### Elección de modelo
# 
# Definamos varias tipologías de modelos de clasificación y evaluemos diferentes métricas utilizando *validación cruzada* y estudiando si la selección de variables mejora o entorpece a los modelos

# In[50]:


from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier


# Añadimos la variable rmse porque usaremos esta función también en el problema de regresión
def compare_models(
    models, 
    X,
    y,
    scoring,
    n_splits=5,
    seed=None, 
    rmse=False, 
    ax=None
):
    # lista de resultados de la validación cruzada y nombres de los algoritmos
    results = []
    names = []
    print('Estadísticas para la métrica {} con CV de {} folds:'.format(scoring,n_splits))
    for name, model in models:
        # generamos los folds
        kfold = model_selection.KFold(n_splits=n_splits, random_state=seed, shuffle=True)
        # realizamos cv
        cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
        names.append(name)
        if not rmse:
            msg = "\t Algoritmo {}: Media = {}, desviación estándar = {}"\
            .format(name, cv_results.mean(), cv_results.std())
        else:
            cv_results = np.sqrt(-cv_results)
            msg = "\t Algoritmo {}: RMSE medio = {}, desviación estándar = {}"\
            .format(name,  cv_results.mean(), cv_results.std())
        print(msg)
        results.append(cv_results)
        
    if ax:
        # Generamos una gráfica de cajas y bigotes para cada algoritmo
        ax.boxplot(results)
        ax.set_xticklabels(names)
    
models = []
models.append(('LR', LogisticRegression(solver='liblinear')))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('GB', GradientBoostingClassifier()))
models.append(('RF', RandomForestClassifier()))
seed = 42


# In[51]:


fig, ax = plt.subplots(1, 2, figsize=(10, 8), sharey=True)

compare_models(
    models, 
    X_train, 
    y_train, 
    scoring="precision", 
    ax=ax[0]
)
ax[0].set_title("Sin selección de variables")

compare_models(
    models, 
    X_train_selected, 
    y_train, 
    scoring="precision", 
    ax=ax[1]
)
ax[1].set_title("Con selección de variables")

fig.suptitle("Comparación de modelos: precisión")


# In[52]:


fig, ax = plt.subplots(1, 2, figsize=(10, 8), sharey=True)

compare_models(
    models, 
    X_train, 
    y_train, 
    scoring="recall", 
    ax=ax[0]
)
ax[0].set_title("Sin selección de variables")

compare_models(
    models, 
    X_train_selected, 
    y_train, 
    scoring="recall", 
    ax=ax[1]
)
ax[1].set_title("Con selección de variables")

fig.suptitle("Comparación de modelos: sensibilidad")


# In[53]:


fig, ax = plt.subplots(1, 2, figsize=(10, 8), sharey=True)

compare_models(
    models, 
    X_train, 
    y_train, 
    scoring="balanced_accuracy", 
    ax=ax[0]
)
ax[0].set_title("Sin selección de variables")

compare_models(
    models, 
    X_train_selected, 
    y_train, 
    scoring="balanced_accuracy", 
    ax=ax[1]
)
ax[1].set_title("Con selección de variables")

fig.suptitle("Comparación de modelos: exactitud balanceada")


# Llegados a este punto, tenemos que preguntarnos **qué métrica queremos maximizar** de cara a que nuestro modelo aporte valor. Como en todo problema de clasificación, siempre hay un tira y afloja entre precisión y sensibilidad, luego no podemos tener un modelo que lo haga bien para ambas métricas. 
# 
# No podemos olvidar que una de las principales aplicaciones de esta clasificación es la **detección de potenciales clientes** con los que vamos a contactar para ofrecerles un depósito en el banco. Por lo tanto, en principio no deberíamos preocuparnos por los falsos negativos (en contraposición con un problema de, digamos, detectar una enfermedad peligrosa). Dejarnos potenciales contratadores atrás no es un problema siempre y cuando **podamos asegurar una precisión alta** en aquellos que elijamos. 
# 
# > Tomaremos la precisión como nuestra métrica de referencia, controlando siempre el resto. En consecuencia, vamos a escoger una regresión logística con selección de variables para realizar tuneado de hiperparámetros 
# 
# Dicho esto, esta decisión dependerá de detalles más concretos del negocio como la capacidad del centro de llamadas

# ## Tuneo de hiperparámetros

# In[54]:


from sklearn.model_selection import GridSearchCV

lr_param_grid = [
    {
        "solver": ["liblinear"],
        "penalty": ["l1", "l2"], 
        "tol": [1e-4, 1e-5], 
        "max_iter": [100, 500]
    }
]
rnd_grid_search = GridSearchCV(
    LogisticRegression(),
    lr_param_grid,
    cv=3,
    scoring="precision",
    return_train_score=True
)
rnd_grid_search.fit(X_train, y_train)


# In[55]:


cvres = rnd_grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(mean_score, params)


# In[56]:


from sklearn.model_selection import GridSearchCV

lr_param_grid = [
    {
        "solver": ["liblinear"],
        "penalty": ["l1", "l2"], 
        "tol": [1e-4, 1e-5], 
        "max_iter": [100, 1000], 
        "fit_intercept": [True, False], 
        "intercept_scaling": [1, 0.1, 10]
    }
]
rnd_grid_search = GridSearchCV(
    LogisticRegression(),
    lr_param_grid,
    cv=3,
    scoring="precision",
    return_train_score=True
)
rnd_grid_search.fit(X_train_selected, y_train)


# In[57]:


cvres = rnd_grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(mean_score, params)


# In[58]:


# Elegimos nuestro clasificador
clf = rnd_grid_search.best_estimator_


# ## Curva de precisión - sensibilidad

# In[59]:


from sklearn.metrics import precision_recall_curve

def plot_precision_recall_threshold(clf, X, y, threshold=.5, ax=None):
    clf_probs = clf.predict_proba(X)[:,1]
    precisions, recalls, thresholds = precision_recall_curve(y,clf_probs)
    # Buscamos el umbral más cercano al dado disponible en thresholds 
    threshold_index = min(range(len(thresholds)), key=lambda i: abs(thresholds[i]-threshold))
    valid_threshold = thresholds[threshold_index] 
    
    # Precision y recall de ese umbral
    precision_threshold = precisions[threshold_index]
    recall_threshold = recalls[threshold_index]
    
    print('Precisión en el umbral {} = %.3f'.format(threshold) % precision_threshold)
    print('Sensibilidad en el umbral {} = %.3f'.format(threshold) % recall_threshold)
    
    if ax:
        # Gráfica
        ax.set_title('Precisión vs Sensibilidad')
        ax.plot(thresholds, precisions[:-1],'b--', label = 'Precisión')
        ax.plot(thresholds, recalls[:-1],'g-', label = 'Sensibilidad')
        ax.plot(valid_threshold, precision_threshold, 'r.', markersize=12,label = 'Umbral = {}'.format(threshold))
        ax.plot(valid_threshold,recall_threshold, 'r.', markersize=12)
        ax.plot([valid_threshold,valid_threshold],[0,max(recall_threshold,precision_threshold)], 'r--')
        ax.plot([0,valid_threshold],[precision_threshold,precision_threshold], 'r--')
        ax.plot([0,valid_threshold],[recall_threshold,recall_threshold], 'r--')
        ax.set_xlabel('Threshold')
        ax.legend()
        ax.set_xlim([0, 1.05])
        ax.grid()


# In[60]:


fig, ax = plt.subplots(figsize=(13, 7))

plot_precision_recall_threshold(clf, X_train_selected, y_train, ax=ax)


# Con el umbral por defecto, tenemos una sensibilidad del 0.17%.  

# ## Testeamos modelo

# In[61]:


def pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = preprocess(df)
    df = process(df)
    df = df[list(rf_selected_features)]
    return df


# In[62]:


X_test = pipeline(bank_test)


# In[63]:


y_test_pred = clf.predict(X_test)


# In[64]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_test_pred)


# In[65]:


precision_score(y_test, y_test_pred)


# In[66]:


recall_score(y_test, y_test_pred)


# Las métricas se mantienen en el test set. 

# ## Guardado de modelos y transformadores

# In[70]:


from joblib import dump

model_path = "model.joblib"
transformer_path = "transformer.joblib"

open(model_path, "w")
dump(clf, model_path)
open(transformer_path, "w")
dump(col_transformer, transformer_path)


# In[71]:


clf.feature_names_in_

