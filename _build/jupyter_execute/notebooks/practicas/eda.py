#!/usr/bin/env python
# coding: utf-8

# (exploratory)=
# # Ejemplo de análisis exploratorio

# Fuente:
#     [Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014
# 
# Descarga: 
#     [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
# 
# Resumen:
#     Los datos provinen de llamadas telefónicas de campañas de marketing de un banco portugués. El objetivo es clasificar si el cliente se subscribirá a un depósito a largo plazo (clasificación binaria)
# 

# ---
# ## Carga de datos

# In[65]:


import os
import urllib
import zipfile

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import scipy.stats as ss

plt.style.use("seaborn")
pd.set_option('display.max_columns', 500)
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")


# In[66]:


working_dir = "."
download_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
bank_path = "data/bank"
raw_data_rel_path = "data/data_raw"
zip_path = "bank-additional/bank-additional-full.csv"
raw_data_path = os.path.join(working_dir, raw_data_rel_path)


# In[67]:


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


# In[68]:


fetch_data(download_url, raw_data_path, bank_path, zip_path)


# In[69]:


bank = load_data(bank_path, zip_path)


# In[70]:


# funciones auxiliares
# variables globales: bank: pd.DataFrame, mask: pd.Series

def create_hist_plot(col: str, ax: plt.Axes, bins: int=None, df: pd.DataFrame=bank) -> None:
    if bins is None:
        bins_total = df[col].max() - df[col].min()
        bins_yes = df.loc[mask][col].max() - df.loc[mask][col].min()
        bins_no = df.loc[~mask][col].max() - df.loc[~mask][col].min()
    else:
        bins_total, bins_no, bins_yes = bins, bins, bins
    ax.hist(x=df[col], bins=bins_total, label="total", align="left")
    ax.hist(x=df.loc[~mask][col], bins=bins_no, label="y=no", alpha=0.5, align="left")
    ax.hist(x=df.loc[mask][col], bins=bins_yes, label="y=yes", alpha=0.5, align="left")
    ax.set_title("Histograma " + col)
    ax.legend()

def create_bar_plot(col: str, ax: plt.Axes, df: pd.DataFrame=bank) -> None:
    ax.bar(
        df.loc[~mask][col].drop_duplicates(),
        df[col].value_counts(dropna=False), 
        label="no"
    )
    ax.bar(
        df.loc[mask][col].drop_duplicates(),
        df.loc[mask][col].value_counts(dropna=False), 
        label="yes"
    )
    ax.tick_params(axis="x", labelrotation=45)
    ax.set_title(col)
    ax.legend()

def describe_num_col(col: str, df: pd.DataFrame=bank) -> pd.DataFrame:
    df_describe = pd.DataFrame(
        [
            df[col].describe(), 
            df.loc[mask][col].describe(),
            df.loc[~mask][col].describe()
        ], 
        index=["total", "y=yes", "y=no"]
    )
    return df_describe

def create_corr_mat_plot(corr_matrix, ax: plt.Axes) -> None:    
    sns.heatmap(corr_matrix, annot=True, ax=ax, linewidths=0.2, fmt=".2f")
    ax.set_title("Matrix de correlación variables numéricas")
        
def cramers_v(x: pd.Series, y: pd.Series) -> np.ndarray:
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))


# In[71]:


bank


# ---
# ## Análisis exploratorio de datos

# Partimos de la siguientes descripciones de lo que represeta cada columna:
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

# Utilizamos el método `info` de pandas para obtener información sobre las columnas del dataset
# - nombre
# - conteo de no nulos
# - tipo

# In[72]:


bank.info()


# Pandas ha detectado un total de 41188 filas, 20 columnas (19 variables regresoras y 1 variable objetivo). Los tipos inferidos han sido 
# - 5 de tipo flotante 
# - 5 de tipo entero 
# - 11 de tipo genérico
# 
# En principio, no ha detectado valores faltantes en ninguna de las variables.

# Realizamos en primer lugar un análisis cuantitativo de las variables de las que disponemos en el dataset. Una vez tengamos claro cómo se estructuran los datos y su calidad, podremos entender mejor el dataset en su conjunto, y consecuentemente diseñar una herramienta de clasificación adecuada. 
# 
# Vamos a estudiar posibles valores nulos, conteos, distribuciones y posibles correlaciones con la variable objetivo.

# ### Variable objetivo: `y`

# In[73]:


mask = bank["y"] == "yes"


# In[74]:


bank["y"].value_counts()


# In[75]:


mask.mean()


# La variable `y` toma dos valores: `"yes"` el **11,26%** de las veces y `"no"` el resto. Trabajamos por lo tanto con **un dataset desbalanceado**.

# ### Variables del cliente: `age`, `job`, `marital`, `education`, `default`, `housing`, `loan`.

# In[76]:


client_cols = ["age", "job", "marital", "education", "default", "housing", "loan"]


# In[77]:


bank[client_cols].info()


# In[78]:


client_num_cols = client_cols[0:1]
client_cat_cols = client_cols[1:]


# Vamos a examinar en primer lugar las variables supuestamente categóricas. Utilizamos el método `value_counts` para realizar conteos.

# In[79]:


n_rows = 2
n_cols = 3
fig, ax = plt.subplots(n_rows, n_cols, figsize=(12, 9))
for i, col in enumerate(client_cat_cols): 
    create_bar_plot(col, ax=ax[i%n_rows][i%n_cols])

fig.suptitle("Variables categóricas de cliente", fontsize=26)
fig.tight_layout()


# Podemos observar que en efecto dichas variables representan categorías de a lo sumo 12 valores únicos. En todas ellas aparece la categoría minoritaria **unknown**, que tiene un peso importante en la variable `default`. Es notorio cómo cambia el porcentaje de instancias con `y=yes` cuando las variables `default`, `job` y `education` toman el valor de `unknown`. 
# 
# Sería interesante llegados a este punto pedir información sobre cómo se generan estos valores `unknown`, ya que a primera vista la distribución de `y` cambia cuando nos restringimos a este segmento del dataset. Podría ser el caso de que estas observaciones provinieran de otra fuente de datos o que el cliente no ha querido dar la información por algún motivo. 

# Calculemos el porcentaje de valores con algún valor de `unknown`

# In[112]:


unknown_mask = (
    bank["default"].isin(["unknown"]) 
    | bank["job"].isin(["unknown"]) 
    | bank["education"].isin(["unknown"]) 
)
unknown_mask.mean()


# Vamos ahora a estudiar la única variable numérica dentro de la categoría de variables de clientes. 

# In[81]:


fig, ax = plt.subplots(figsize=(6, 4))
create_hist_plot("age", ax=ax)
fig.tight_layout()


# Vemos valores de edad desde los 17 años hasta más de los 90, con un salto importante en la edad de jubilicación (en torno a los 60). Estudiemos media y algunos percentiles de la edad segmentando por positivos y negativos de la variable objetivo `y`.

# In[82]:


describe_num_col("age")


# Tanto media como percentiles son parecidos, las observaciones positivas tienen un poco más de varianza.

# ### Variables relacionadas con el contacto con el cliente: `contact`, `month`, `day_of_week`, `duration`

# In[83]:


contact_cols = ["contact", "month", "day_of_week", "duration"]


# In[84]:


bank[contact_cols].info()


# In[85]:


contact_cat_cols = contact_cols[:-1]
contact_num_cols = contact_cols[-1:]


# In[86]:


n_rows = 3
fig, ax = plt.subplots(1, n_rows, figsize=(15, 5))
for i, col in enumerate(contact_cat_cols): 
    create_bar_plot(col, ax=ax[i])

fig.suptitle("Variables categóricas de contacto", fontsize=26)
fig.tight_layout()


# Notamos que 
# - El método de contacto mayoritario es el móvil, casi con el doble de contactos. 
# - Los meses de mayo, julio, agosto y junio son los que tienen mayor actividad durante las campañas
# - El número de contactos por día de la semana es bastante uniforme
# - La distribución de `y` **cambia radicalmente para los meses de diciembre, marzo, abril y septiembre**. Sería interesante saber por qué el número de contactos decrece en función de la época del año (campañas de marketing). Podría ser que durante los meses de no campaña, se contactará sólo con el cliente una vez el mismo se haya interesado, de ahí una tasa de positivos tan alta.

# En cuanto a la variable `duration` obtenemos el siguiente histograma 

# In[87]:


fig, ax = plt.subplots(figsize=(6, 4))

create_hist_plot("duration", bins=50, ax=ax)
fig.tight_layout()


# In[88]:


describe_num_col("duration")


# Vemos que la media del tiempo de llamada cuando el cliente acaba contratando el depósito, como era de esperar. 
# 
# A pesar de ser una variable con un potencial prometedor para nuestra tarea, no podemos olvidar que si nuestro objetivo es utilizar el modelo para filtrar potenciales clientes y crear un listado que se le pase a un *call center*, **nunca vamos a disponer de `duration`**, ya que se informa su valor a futuro, una vez el listado está hecho. El resto de variables sobre el contacto (y resto de agrupaciones) sí estarían potencialmente disponibles a la hora de crear un listado.

# ### Variables del historial de contactos: `campaign`, `pdays`, `previous`, `poutcome`.

# In[89]:


history_cols = ["campaign", "pdays", "previous", "poutcome"]


# In[90]:


bank[history_cols].info()


# #### `poutcome`

# In[91]:


fig, ax = plt.subplots()
create_bar_plot("poutcome", ax=ax)

fig.tight_layout()


# Observamos que la tasa de positivos es mucho mayor en clientes que ya han contratado un depósito a largo plazo. Veamos cuántos han repetido 

# In[92]:


bank[bank["poutcome"] == "success"]["y"].value_counts()


# Vamos con las variables numéricas

# #### `pdays`

# In[93]:


describe_num_col("pdays")


# In[94]:


bank["pdays"].value_counts(normalize=True).iloc[:3]


# Vemos que la variable `pdays` tiene la mayoría de sus valores en `999` (> 96%), que se corresponde con que el cliente no ha sido contactado en la anterior campaña. Veamos que pinta tiene el histograma del aquellas observaciones que sí tengan registro en `pdays`

# In[95]:


fig, ax = plt.subplots()
create_hist_plot("pdays", ax=ax, df=bank[bank["pdays"] < 999])


# Se puede apreciar que el cliente es contactado de nuevo tras 3 o 6 días en la misma campaña. Aquellos a los que se contacta varias veces parecen tener una tasa de positivos alta 

# In[96]:


bank[bank["pdays"] < 999]["y"].value_counts()


# Parece una buena idea crear una variable binaria a partir de `pdays` que indique si el cliente ha sido contactado con anterioridad en otra campaña (`pdays < 999`).

# #### `previous`

# Exploremos `previous`, que se define como `campaign` pero cuenta contactos realizados *antes* de la campaña actual

# In[97]:


describe_num_col("previous")


# In[98]:


# campaign
fig, ax = plt.subplots()
create_hist_plot("previous", ax=ax, df=bank[bank["previous"] < 5])


# Destaca que el valor medio de `previous` en los positivos es más del doble que en los negativos. 

# Llegados a este punto sería interesante saber si la información de `poutcome`, `previous`, `pdays` es coherente, en el sentido de que si `poutcome = "nonexistent"`, es decir, el cliente no ha participado en campañas anteriores, entonces `previous` tiene que valer 1 y `pdays` debe ser 999.

# In[99]:


bank[["poutcome", "previous", "pdays"]].value_counts(normalize=True).sort_values(ascending=False)


# Vemos que dicha hipótesis se cumple, sim embargo `pdays` puede valer 999 incluso cuando `poutcome` y `previous` están informados. Parece que las tres variables codifican información similar, luego sería interesante ver la correlación con la variable objetivo y elegir una de ellas. Dado que ambas son categóricas, utilizamos el [coeficiente V de Cramér](https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V)

# In[100]:


cols = ["poutcome", "previous", "pdays"]
pd.DataFrame(
    [cramers_v(bank[col], bank["y"]) for col in cols], 
    index=cols, 
    columns=["V de Cramér"]
)


# Tendría sentido por lo tanto quedarnos con `poutcome` o con `pdays`.

# #### `campaign`

# In[101]:


describe_num_col("campaign")


# El mínimo de la variable es 1, por lo que deducimos que **el contacto que se está informando en la fila se cuenta**. Más de 75% tiene menos de 3 contactos en la campaña. Notemos también que **la media de contactos en las personas que acaban contratando es menor**.

# In[102]:


# campaign
fig, ax = plt.subplots()
create_hist_plot("campaign", ax=ax, df=bank[bank["campaign"] < 10])


# In[103]:


cramers_v(bank["campaign"], bank["y"])


# ### Variables socioeconómicas: `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`

# In[104]:


economic_cols = ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]


# In[105]:


bank[economic_cols].info()


# In[106]:


for col in economic_cols:
    print(f"Número de valores únicos en {col}: {bank[col].nunique()}")


# Podemos observar que 
# - `emp.var.rate` y `nr.employed` son indicadores cuatrimestrales
# - `cons.price.idx` y `cons.conf.idx` son anuales
# - `euribor3m` es diario
# 
# Deducimos que el histórico se corresponde con 2 años y algunos meses, aunque como hemos visto los meses de enero y febrero no tienen registros.

# ## Correlaciones

# ### Entre variables numéricas

# Calculamos el [coeficiente de correlación de Pearson](https://es.wikipedia.org/wiki/Coeficiente_de_correlaci%C3%B3n_de_Pearson) entre cada par de atributos numéricos

# In[107]:


bank["y_num"] = mask
corr_matrix = bank.corr()
corr_matrix


# In[108]:


fig, ax = plt.subplots(figsize=(8, 7))

create_corr_mat_plot(corr_matrix, ax=ax)


# Destacan
# - Una correlación positiva entre algunas de las variables socioeconómicas. 
# - La variable `previous` tiene una correlación negativa con las variables socioeconómicas. Destaca el coeficiente negativo del par `previous`-`pdays`

# ### Variables categóricas 

# Utilizamos de nuevo el coeficiente V de Cramér

# In[109]:


cat_cols = bank.select_dtypes("object").columns.drop("y")

v = dict()
for col in cat_cols:
    v[col] = cramers_v(bank[col], bank["y"])


# In[110]:


cramers_v(bank["loan"], bank["y"])


# In[111]:


v = pd.Series(v).sort_values()
v

