#!/usr/bin/env python
# coding: utf-8

# # Construyendo un modelo de regresión lineal

# En esta sección vamos a utilizar un dataset sobre el precio de la vivienda en el estado de California para construir un modelo de regresión con la librería `sklearn` haciendo uso de sus clases predefinidas para modelos y procesamiento, apoyándonos como es usual en `numpy`, `pandas` y `matplotlib`. 

# ## Imports y configuraciones 
# 
# En primer lugar, vamos a importar algunos módulos, configurar cómo se renderizan las figuras de matplotlib para que se vean bien en la página y hacer algunas comprobaciones sobre el versionado de las librerías utilizadas. Definimos una función para guardar imágenes en caso de que queramos persisitir alguna de ellas.

# In[ ]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")


# In[ ]:


# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import os

# To plot pretty figures

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams["figure.figsize"] = (12, 8)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# ## Descargando los datos 

# La siguiente función descarga y descomprime los datos de viviendas desde una URL específica a una ruta específica en el sistema de archivos. Si la ruta no existe, la función la crea. Después, descarga un archivo .tgz de la URL especificada y lo descomprime en la ruta especificada.

# In[ ]:


import os
import tarfile
import urllib.request

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# In[ ]:


fetch_housing_data()


# Ahora utilizamos la función `pd.read_csv` de pandas para leer un `.csv` cuya ruta hay que especificar 

# In[ ]:


import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# In[ ]:


housing = load_housing_data()
housing.head()


# Echemos un vistazo rápido a la información disponible sobre este dataframe

# In[ ]:


housing.info()


# Tenemos 9 columnas numéricas, una columna categórica y un total de 20640 filas. **Nuestro objetivo será predecir la variable `median_house_value`** cuando dispongamos del resto de columnas para un barrio nuevo. Veamos qué posibles valores tiene la columna tipo `object` llamada `ocean_proximity`

# In[ ]:


housing["ocean_proximity"].value_counts()


# Vamos con el resto de columnas numéricas 

# In[ ]:


housing.describe()


# Parece razonable proponer el siguiente ejercicio 

# :::{exercise}
# :label: regression-model-1
# 
# Convierte la columna `ocean_proximity` en categórica y las columnas susceptibles de ser de tipo entero. Recuerda que las columnas que tienen valores faltantes no pueden ser convertidas en tipo entero. 
# 
# :::

# Vamos a representar el historgrama de las variables numéricas de nuestro dataframe. Para ello basta llamar al método `hist` de `pd.DataFrame`

# In[ ]:


housing.hist(bins=50, figsize=(20,15))
plt.show()


# :::{exercise}
# :label: regression-model-2
# 
# ¿Qué conclusiones puedes sacar de estos datos? ¿Cómo podemos proceder?
# 
# :::

# ## Creando un conjunto de test

# La siguiente función hace una partición entre conjunto de entrenamiento y test de forma similar a `train_test_split` de `sklearn.model_selection`.

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# In[ ]:


train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set))
print(len(test_set))


# :::{exercise}
# :label: regression-model-3
# 
# ¿Qué problemas puede tener una partición de este tipo? ¿Cómo podemos solventarlo? Véase [`StratifiedShuffleSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html)
# 
# :::

# Debemos dejar el conjunto de test apartado antes de hacer cualquier transformación y una vez hayamos concluido la tarea de desarrollar modelos de predicción, podremos usarlo para evaluar dichos modelos. Antes de seguir, vamos a visualizar los datos de los que disponemos para obtener más información de cara a diseñar procesamiento y entrenamiento de modelos. 

# In[ ]:


housing = train_set.copy()


# ## Visualizando los datos

# Realicemos un gráfico de tipo scatter para representar la información geográfica (longitud y latitud) del dataset.

# In[ ]:


housing.plot(kind="scatter", x="longitude", y="latitude")
plt.show()


# Vamos a incluir un poco de transparencia en los puntos para poder apreciar las zonas con mayor densidad 

# In[ ]:


housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
plt.show()


# Ahora añadimos opciones para que el tamaño de cada punto venga en función de la población del distrito y añadimos una barra de color para el precio mediano (variable objetivo)

# In[ ]:


housing.plot(
    kind="scatter", 
    x="longitude", 
    y="latitude", 
    alpha=0.4,
    s=housing["population"]/100,
    label="population",
    c="median_house_value", 
    cmap=plt.get_cmap("jet"), 
    colorbar=True,
    sharex=False
)
plt.legend()
plt.show()


# A continuación descargamos una imagen del mapa de california 

# In[ ]:


# Download the California image
images_path = os.path.join(PROJECT_ROOT_DIR, "images", "end_to_end_project")
os.makedirs(images_path, exist_ok=True)
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
filename = "california.png"
print("Downloading", filename)
url = DOWNLOAD_ROOT + "images/end_to_end_project/" + filename
urllib.request.urlretrieve(url, os.path.join(images_path, filename))


# In[ ]:


import matplotlib.image as mpimg

california_img=mpimg.imread(os.path.join(images_path, filename))
housing.plot(
    kind="scatter", 
    x="longitude", 
    y="latitude", 
    s=housing['population']/100, 
    label="Population",       
    c="median_house_value", 
    cmap=plt.get_cmap("jet"),         
    colorbar=False, 
    alpha=0.4
)
plt.imshow(
    california_img, 
    extent=[-124.55, -113.80, 32.45, 42.05], 
    alpha=1,
    cmap=plt.get_cmap("jet")
)
plt.ylabel("Latitude", fontsize=14)
plt.xlabel("Longitude", fontsize=14)

prices = housing["median_house_value"]
tick_values = np.linspace(prices.min(), prices.max(), 11)
cbar = plt.colorbar(ticks=tick_values/prices.max())
cbar.ax.set_yticklabels([f"${round(v/1000)}k" for v in tick_values], fontsize=14)
cbar.set_label('Median House Value', fontsize=16)

plt.legend(fontsize=16)
# save_fig("california_housing_prices_plot")
plt.show()


# :::{exercise}
# :label: regression-model-4
# 
# Saca conclusiones del anterior gráfico.
# 
# :::

# ## Profundizando en los datos 
# 
# ### Correlaciones 
# Vamos a calcular la matriz de correlaciones de las variables numéricas del dataframe

# In[ ]:


corr_matrix = housing.corr()


# Veamos cómo de correlacionan las variables regresoras con la objetivo 

# In[ ]:


corr_matrix["median_house_value"].sort_values(ascending=False)


# La función `heatmap` de la librería **seaborn** (basada en matplotlib) nos puede servir para visualizar la matriz de correlaciones

# In[ ]:


import seaborn as sns

def create_corr_mat_plot(corr_matrix, ax: plt.Axes) -> None:    
    sns.heatmap(corr_matrix, annot=True, ax=ax, linewidths=0.2, fmt=".2f")
    ax.set_title("Matrix de correlación variables numéricas")


# In[ ]:


fig, ax = plt.subplots()
create_corr_mat_plot(corr_matrix, ax)
plt.show()


# También podemos usar la función `scatter_matrix` de `pandas.plotting` para hacer un scatter plot de los atributos más correlacionados con la variable objetivo 

# In[ ]:


from pandas.plotting import scatter_matrix

attributes = (
    corr_matrix["median_house_value"]
    .sort_values(ascending=False)
    .iloc[:4]
    .index
)
scatter_matrix(housing[attributes], figsize=(12, 8))
plt.show()


# Hagamos zoom en el plot de la variable más correlacionada con `median_house_value`

# In[ ]:


housing.plot(
    kind="scatter", 
    x="median_income", 
    y="median_house_value",
    alpha=0.1
)
plt.axis([0, 16, 0, 550000])
plt.show()


# Vemos como efectivamente hay una correlación positiva entre ambas variables y el corte en el valor mediano de las viviendas en 500.000 que habíamos notado en el histograma. Pero también hay un par de líneas horizontales cerca de los 350.000 y 450.000 que no habíamos notado antes, ni tampoco se visualizan en el resto de gráficas de tipo scatter.

# In[ ]:


housing.info()


# ## Generación de nuevas variables
# 
# Un paso muy importante en el procesamiento de datos que alimentan a un modelo es la combinación de atributos para generar nuevas variables que puedan mejorar el rendimiento del mismo.
# 
# En este caso por ejemplo tenemos el número total de habitaciones en un distrito (variable `total_rooms`) y también el número total de cuartos `total_bedrooms`, pero parece razonable que otros atributos, como el número de habitaciones por casa o el ratio de cuartos por habitación podrían tener más sentido. Vamos a definirlos  

# :::{exercise}
# :label: regression-model-5
# 
# Define las siguientes columnas de `housing` 
# 
# - `rooms_per_household`: número de habitaciones por vivienda
# - `bedrooms_per_room`: número de cuartos por habitación
# - `population_per_household`: habitantes por vivienda
# 
# :::

# :::{solution} regression-model-5
# :class: dropdown
# 
# ```
# housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
# housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
# housing["population_per_household"]=housing["population"]/housing["households"]
# ```
# 
# :::
# 

# In[ ]:


housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]


# In[ ]:


corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# Parece que las variables `rooms_per_household` y `bedrooms_per_room` tienen grados de correlación superiores a los de las variables originales. 

# ## Preparando los datos para un modelo de ML
# 
# Una vez realizado el análisis de los datos, el procesado de los mismos para alimentar nuestro modelo de ML podría constar de las siguientes partes 
# 
# - Limpieza de datos 
# - Codificación de variables de texto y categóricas 
# - Ingeniería de variables 
# - Escalado de variables
# 
# Esta secuencia de procesos puede cambiar en función de las particularidades de nuestros datos y el objetivo del modelo. Vamos a ir tratando una a una, pero antes es importante dejar nuestra variable objetivo desacoplada del procesamiento 

# In[ ]:


housing = train_set.drop("median_house_value", axis=1) 
housing_labels = train_set["median_house_value"].copy()


# A continuación vamos a separar las variables numéricas de las categóricas 

# In[ ]:


num_cols = housing.select_dtypes("number").columns
cat_cols = housing.select_dtypes(exclude="number").columns


# ### Limpieza de datos 
# 
# Básicamente se trata de lidiar con los datos faltantes o defectuosos. En nuestro caso vamos a **imputar** los valores faltantes de la única columna que parece tenerlos, `total_bedrooms`. 

# In[ ]:


median = housing["total_bedrooms"].median() 
housing["total_bedrooms"].fillna(median, inplace=True)


# :::{exercise}
# :label: regression-model-6
# 
# Utiliza la clase `SimpleImputer` de `sklearn.impute` para relizar esta tarea.
# 
# :::

# ### Transformando datos categóricos
# 
# Vamos a realizar un [*one-hot-encoding*](https://es.wikipedia.org/wiki/One-hot) para convertir en numúrica la única variable categórica que tenemos en nuesta dataset, `ocean_proximity`. Para ello vamos a usar la clase `OneHotEncoder` de `sklearn.preprocessing`. 

# In[ ]:


from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing[cat_cols])
housing_cat_1hot


# Por defecto, `OneHotEncoder` devuelve una array disperso o *sparse*, pero podemos convertirlo en un array convencional llamando al método `toarray`

# In[ ]:


housing_cat_1hot.toarray()


# ### Ingeniería de variables
# 
# Usando las clases `BaseEstimator` y `TransformerMixin` de `sklearn.base`, podemos crear una clase que transforma nuestro dataset para añadir los atributos que durante el análisis hemos visto que podrían ser útiles 

# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin

col_names = "total_rooms", "total_bedrooms", "population", "households"
rooms_ix, bedrooms_ix, population_ix, households_ix = [
    housing.columns.get_loc(c) for c in col_names] 

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self  # es obligatorio definir este método
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder()
housing_extra_attribs = attr_adder.transform(housing.values)


# `attr_adder.transform` devuelve un array (más adelante veremos que esto es lo óptimo), pero podemos recuperar el dataframe fácilmente

# In[ ]:


housing_extra_attribs = pd.DataFrame(
    housing_extra_attribs,
    columns=list(housing.columns)+["rooms_per_household", "population_per_household"],
    index=housing.index)
housing_extra_attribs.head()


# ### Escalado de variables 
# 
# Podemos usar varias estrategias para escalar nuestras variables, por ejemplo la clase `StandardScaler` normaliza nuestra variable como 
# 
# $$
# Z = \frac{X - \mu}{\sigma} 
# $$
# 
# o `MinMaxScaler` escala la variable para que encaje con un intervalo dado
# 
# $$
# Z = a + \frac{X - \text{min}(X)}{\text{max}(X) - \text{min}(X)}(b-a)
# $$

# ## Definiendo un pipeline
# 
# Vamos a organizar y condensar las transformaciones hechas en un objeto de tipo `sklearn.pipeline`

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# pipeline numérico
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])


# Podemos combinar pipelines que se aplican a diferentes columnas con `ColumnTransformer`

# In[ ]:


from sklearn.compose import ColumnTransformer

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", OneHotEncoder(), cat_cols),
    ])

housing_prepared = full_pipeline.fit_transform(housing)


# In[ ]:


housing_prepared.shape


# ## Entrenamiento y evaluación
# 
# Vamos a utilizar un regresor lineal como el que construimos manualmente en otra práctica, pero en este caso importamos directamente desde `sklearn.linear_model`

# In[ ]:


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)


# Esto sería un ejemplo de aplicación del pipeline de procesmiento completo con su correspondiente predicción 

# In[ ]:


some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

print("Predictions:", lin_reg.predict(some_data_prepared))


# Comparamos con los valores reales 

# In[ ]:


print("Labels:", list(some_labels))


# Podemos obtener el error cuadrático medio de nuestro modelo con `sklearn.metrics.mean_squared_error`

# In[ ]:


from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# In[ ]:


housing_labels.mean()


# Es mejor opción realizar un **cross-validation** para hacernos una idea del error de nuestro modelo en test.  

# In[ ]:


from sklearn.model_selection import cross_val_score

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

lin_scores = cross_val_score(
    lin_reg,
    housing_prepared, 
    housing_labels,
    scoring="neg_mean_squared_error", 
    cv=10
)

lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)


# Vamos a comparar la performance con un random forest 

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(housing_prepared, housing_labels)

forest_scores = cross_val_score(
    forest_reg,
    housing_prepared, 
    housing_labels,
    scoring="neg_mean_squared_error", 
    cv=10
)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)


# ## Buscando el mejor modelo tuneando los hiperparámetros
# 
# La idea es ahora ir modificando los hiperparámetros de `RandomForestRegressor`, por ejemplo 
# 
# - `n_estimators`: número de árboles en el bosque
# - `max_features`: máximo número de variables usadas para construir las particiones en los árboles
# - `boostrap`: si las filas seleccionadas para construir un árbol se descartan. 
# 
# Hay varias estrategias para construir un espacio de hiperparámetros. Una de ellas es crear una rejilla de hiperparámetros probando todas las posibles combinaciones, para ello podemos usar la clase `GridSearchCV` de `sklearn.model_selection`

# Vamos a especificar un argumento `scoring="neg_mean_squared_error"`, ya que la clase nos dará el mejor estimador en función a esta métrica (siempre intenando maximizarla)

# In[ ]:


from sklearn.model_selection import GridSearchCV

param_grid = [
    # Intentamos 12 (3×4) combinaciones de hiperparámetros en primer lugar
    {'n_estimators': [3, 10, 100], 'max_features': [2, 4, 6, 8]},
    # Otras 6 combinaciones después
    {'bootstrap': [False], 'n_estimators': [3, 100], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)

# cross validation con 5 folds
grid_search = GridSearchCV(
    forest_reg, 
    param_grid, cv=5,
    scoring='neg_mean_squared_error',
    return_train_score=True
)
grid_search.fit(housing_prepared, housing_labels)


# In[ ]:


cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

print("\n")
print("Best predictor: ")
print(np.sqrt(-grid_search.best_score_), grid_search.best_params_)


# ## Evaluar el conjunto de test
# 
# El último paso que nos queda es evaluar el conjunto de test que reservamos al principio del análisis

# In[ ]:


final_model = grid_search.best_estimator_

X_test = test_set.drop("median_house_value", axis=1)
y_test = test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)


# In[ ]:


final_rmse


# Parece razonable descartar que nuestro modelo esté sobreentrenado. Podemos guardar todo el procesmiento y el modelo en un único objeto de nuevo usando `Pipeline`

# In[ ]:


full_pipeline_with_predictor = Pipeline([
        ("preparation", full_pipeline),
        ("final_model", final_model)
    ])

full_pipeline_with_predictor.fit(housing, housing_labels)
full_pipeline_with_predictor.predict(some_data)


# Finalmente, tendremos que guardar nuestro transformador en disco para ser consumido posteriormente, para ello usamos la librería estándar `joblib`

# In[ ]:


import joblib
joblib.dump(full_pipeline_with_predictor, "full_pipeline_with_predictor.pkl")


# In[ ]:


full_pipeline_with_predictor_copy = joblib.load("full_pipeline_with_predictor.pkl") 


# In[ ]:


type(full_pipeline_with_predictor_copy)

