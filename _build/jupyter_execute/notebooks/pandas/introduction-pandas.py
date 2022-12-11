#!/usr/bin/env python
# coding: utf-8

# (pandas)=
# # Pandas

# En esta sección haremos una introducción a la librería [pandas](https://pandas.pydata.org/) de Python, una herramienta muy útil para **el análisis de datos**. Proporciona estructuras de datos y operaciones para manipular y analizar datos de manera rápida y eficiente, así como funcionalidades de lectura y escritura de datos en diferentes formatos, como CSV, Excel, SQL, entre otros. También permite realizar operaciones matemáticas y estadísticas en los datos, así como visualizarlos en gráficos y tablas de manera cómoda gracias a su integración con **numpy** y **matplotlib**. En resumen, pandas es una librería muy útil para cualquier persona que trabaje con datos y necesite realizar análisis y operaciones en ellos de manera rápida y eficiente.

# <div style="display: flex; align-items: center; justify-content: center;">
#     <img src="https://drive.google.com/uc?id=1HTFx_ZaV6QywEjp_6Dd_NVe78L7oyzsX"/>
# </div>

# La integración entre numpy y pandas se realiza mediante el uso de los arrays de numpy como el tipo de dato subyacente en las estructuras de datos de pandas. Esto permite que pandas utilice la eficiencia y la velocidad de cálculo de numpy en sus operaciones, mientras que proporciona una interfaz de usuario más amigable y especializada para trabajar con datos tabulares.

# Normalmente el módulo se suele importar con el alias `pd`

# In[ ]:


import pandas as pd
import numpy as np


# ## Series

# Una **serie** de pandas es una estructura de datos unidimensional, junto con una secuencia de etiquetas para cada dato denominada **índice**. Podemos crear una serie de pandas a través de una lista de Python 

# In[ ]:


s = pd.Series([4, 7, -5, 3])
s


# En este ejemplo vemos que pandas asigna por defecto un índice numérico que etiqueta los datos que le hemos pasado mediante la lista. Pandas gestiona estos datos como un array de numpy, que es accesible mediante el atributo `values`. También observamos que el tipo de numpy elegido ha sido `int64`

# In[ ]:


s.values


# El índice está disponible en el atributo `index`. En este caso crea un objeto similar al `range` de Python, pero más generalmente serán instancias de `pd.Index`

# In[ ]:


s.index


# <div style="display: flex; align-items: center; justify-content: center;">
#     <img src="https://drive.google.com/uc?id=17YVcnJlr72tSlmnyF8inGCNPXw-nUMD4"/>
# </div>

# Podemos proporcionar un índice cuando creemos la serie

# In[ ]:


s2 = pd.Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])


# In[ ]:


s2.index


# También podemos añadirle un atributo `name` tanto `pd.Series` como a un índice

# In[ ]:


s2.name = "test"
s2.index.name = "letras"
s2


# La noción de índice en pandas generaliza en cierto sentido los índices de numpy. Igual que en numpy, podemos acceder a los elementos de la series a través del índice y modificarlos

# In[ ]:


s2["a"]


# In[ ]:


s2["a"] = 6
s2


# Podemos tambíen indicar una subserie

# In[ ]:


s2[['c', 'a', 'd']]


# Las operaciones que estarían disponibles sobre el array subyacente a la serie se pueden aplicar directemante a la misma 

# In[ ]:


s2[s2 > 5]


# In[ ]:


s2*2


# In[ ]:


np.exp(s2)


# De hecho, una manera muy frecuente de crear una serie es a partir de un diccionario. Las claves se ordenarán y formarán el índice de la serie, como en el siguiente ejemplo:

# In[ ]:


sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
s3 = pd.Series(sdata)
s3


# Si queremos introducir un orden específico entre las claves del diccionario, entonces podemos combinar el pasar el diccionario junto con la lista de etiquetas ordenadas:

# In[ ]:


states = ['California', 'Ohio', 'Oregon', 'Texas']
s4 = pd.Series(sdata, index=states)
s4


# Nótese que solo se incluye en el índice lo incluido en la lista (por ejemplo `Utah` no forma parte del índice a pesar de que es una clave del diccionario). Como `California` no es una clave del diccionario, pero se ha incluido en el índice, se incluye con valor `NaN`(*Not a Number*), que es la manera en pandas para indicar valores inexistentes. 

# Con `isnull` podemos localizar qué entradas de la serie tienen valores inexistentes:

# In[ ]:


s4.isnull()


# En las series, como con los arrays de numpy, podemos realizar operaciones vectorizadas. Lo interesante aquí es que las operaciones se *alinean* por las correspondientes etiquetas. Por ejemplo:

# In[ ]:


s3 + s4


# Los tipos que solemos manejar en las series de pandas son similares que los de numpy, aunque existe un tipo particular de pandas bastante útil, que nos permite usar funcionalidades y ahorrar memoria, el tipo `category`

# In[ ]:


s = pd.Series(
    ["s", "m", "l", "xs", "xl"], 
    dtype="category"
)
s


# :::{exercise}
# :label: pandas-series
# 
# Carga las series `city_mpg` y `highway_mpg` con el siguiente código
# ```
# url = "https://github.com/mattharrison/datasets/raw/master/data/vehicles.csv.zip"
# df = pd.read_csv(url)
# city_mpg = df.city08
# highway_mpg = df.highway08
# ``` 
# - ¿Cuántos elementos hay en las series? ¿De qué tipo son? 
# - Calcula el mínimo, el máximo y la mediana de la Serie utilizando las funciones `min`, `max` y `median` respectivamente.
# - Utiliza la función `pd.cut` para dividir la Serie de precios en cuatro categorías: "bajo", "medio-bajo", "medio-alto" y "alto", utilizando los cuartiles como límites de las categorías.
# - Cuenta el número de elementos en cada categoría utilizando la función `value_counts`.
# - Realiza un histograma y un gráfico de barras.
# 
# :::

# ## DataFrames

# Un **DataFrame** de pandas es una tabla bidimensional, con las columnas y las filas en un determinado orden. Cada columna puede ser de un tipo diferente. En términos de índices: tanto las filas como las columnas están indexadas. 
# 
# > Puede ver un DataFrame como un diccionario en el que las claves son las etiquetas de las columnas, y todos los valores son **Series** de pandas que comparten el mismo índice. 
# 
# Aunque hay muchas maneras de crear un *DataFrame*, una de las más frecuentes es mediante un diccionario cuyos valores asociados a las claves son listas de la misma longitud. Por ejemplo:   

# In[ ]:


data = {
    'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
    'year': [2000, 2001, 2002, 2001, 2002, 2003],
    'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]
}
frame = pd.DataFrame(data)


# Nótese la forma en que se muestra un *DataFrame* en Jupyter:

# In[ ]:


frame


# Podemos obtener infomación del DataFrame con el método `info`

# In[ ]:


frame.info()


# Cuando los *DataFrames* son grandes, puede ser útil el método `head`, que muestar solo las primeras $n$ filas de la tabla:

# In[ ]:


frame.head()


# Como en el caso de las Series, podemos proporcionar las columnas en un orden determinado: 

# In[ ]:


pd.DataFrame(data, columns=['year', 'state', 'pop'])


# Y también podemos indicar expresamente el índice de las filas:

# In[ ]:


pd.DataFrame(
    data, 
    columns=['pop', 'year', 'state'], 
    index=['one', 'two', 'three', 'four', 'five', 'six']
)


# Si al proporcionar los nombres de las columnas damos una de ellas que no aparece en el diccionario con los datos, entonces se crea la columnos con valores no determinados:

# In[ ]:


df2 = pd.DataFrame(
    data, 
    columns=['year', 'state', 'pop', 'debt'],
    index=['one', 'two', 'three', 'four', 'five', 'six']
)
df2


# Los atributos `index` y `columns` nos devuelven los correspondientes índices de las filas y de las columnas (ambos son objetos `Index` de pandas): 

# In[ ]:


df2.index


# In[ ]:


df2.columns


# Para acceder a una columna en concreto del *DataFrame*, podemos hacerlo usando la notación de diccionario, o también como atributo. En ambos casos se devuelve la correspondiente columna como un objeto *Series* de pandas: 

# In[ ]:


df2['state']


# In[ ]:


df2.year


# De igual manera, podemos acceder a una fila del *DataFrame* mediante el método `loc`, que veremos con más detenimiento en lo siguiente. La fila también se devuelve como un objeto *Series*, cuyo índice está formado por los nombres de las columnas:

# In[ ]:


df2.loc['three']


# Veamos ahora ejemplos sobre cómo podemos modificar columnas mediante asignaciones. En general, muchas de los procedimientos de numpy aquí también son válidos, pero teniendo en cuenta que indexamos mediante el nombre de la columna: 

# Por ejemplo, asignar el mismo valor a toda una columna:

# In[ ]:


df2['debt'] = 16.5
df2


# O asignar mediante una secuencia:

# In[ ]:


df2['debt'] = np.arange(6.)
df2


# Cuando a una columna le asignamos una lista o un array, como en el ejemplo anterior, la longitud de la secuencia debe de coincidir con el número de filas del *DataFrame*. Sin embargo, podemos asignar con un objeto *Series* y los valores se asignarán alineando por el valor del índice, incluso parcialmente (al resto se el asignará *NaN*):

# In[ ]:


val = pd.Series(
    [-1.2, -1.5, -1.7], 
    index=['two', 'four', 'five']
)
df2['debt'] = val
df2


# Si asignamos una columna que no existe, ésta se creará. Por ejemplo:

# In[ ]:


df2['eastern'] = df2.state == 'Ohio'
df2


# Podemos borrar una colunma con el método `drop`

# In[ ]:


df2.drop(columns="eastern", inplace=True)
# alternativa -> del df2['eastern']
df2.columns


# In[ ]:


df2


# Otra forma de crear un *DataFrame* es a partir de un diccionario de diccionarios, en el que las claves externas constituyen las etiquetas de las columnas, y las internas como las de las filas:

# In[ ]:


pop = {
    'Nevada': {2001: 2.4, 2002: 2.9},
    'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}
}


# In[ ]:


df3 = pd.DataFrame(pop)
df3


# Como en numpy, podemos también obtener la traspuesta de un *DataFrame*, quedando las filas como columns y viceversa:

# In[ ]:


df3.T


# También se puede dar un *DataFrame* como un diccionario en el que cada clave (columna) tiene asociada una serie: 

# In[ ]:


pdata = {
    'Ohio': df3['Ohio'][:-1],
    'Nevada': df3['Nevada'][:2]
}
pd.DataFrame(pdata)


# Con el atributo `name` (tanto de `index`como de `columns`) podemos acceder y/o modificar el nombre de las filas y las columnas, que se mostrarán al mostrarse la tabla:

# In[ ]:


df3.index.name = 'year'
df3.columns.name = 'state'
df3


# Por último, mediante `values`, accedemos a un array bidimensional con los valores de cada entrada de la tabla:

# In[ ]:


df3.values


# In[ ]:


df2.values # el dtype se acomoda a lo más general. 


# ### Resumen de algunas maneras de crear un *DataFrame*:
# 
# * Array bidimensional, opcionalmente con `index` y/o `columns`
# * Diccionario de arrays, listas o tuplas de la misma longitud; cada clave se refiere a una columna
# * Diccionario de *Series*; cada clave es una columna y las filas se alinean según los índices de las series, o bien se le pasa explícitamente el índice. 
# * Diccionario de diccionarios: las claves externas son las columnas, las internas las filas.
# * Lista de listas o tuplas: como en el caso de array bidimensional. 

# :::{exercise}
# :label: pandas-create-df
# 
# Crea un DataFrame de 5 filas y columnas `Nombre`, `Edad`, `Peso` con alguno de los métodos mencionados arriba. 
# 
# :::

# ## Funcionalidades básicas

# ### Eliminando entradas de un eje

# Mediante `drop`, podemos *crear nuevos objetos* resultantes de eliminar filas o columnas completas. Veamos algunos ejemplos. En primer lugar, con las *Series*:

# In[ ]:


s = pd.Series(
    np.arange(5.), 
    index=['a', 'b', 'c', 'd', 'e']
)
s


# In[ ]:


new_s = s.drop('c') # notemos que inplace=False (valor por defecto)
new_s


# In[ ]:


# varias entradas a la vez
s.drop(['d', 'c'])


# Ahora veamos el uso de `drop` con *DataFrames*:

# In[ ]:


data = pd.DataFrame(
    np.arange(16).reshape((4, 4)),
    index=['Ohio', 'Colorado', 'Utah', 'New York'],
    columns=['one', 'two', 'three', 'four']
)
data


# Por defecto, se eliminan del eje 0 (las filas):

# In[ ]:


data.drop(['Colorado', 'Ohio'])


# Podemos eliminar columnas, indicándo que se quiere hacer en `axis=1` o `axis='columns'`:

# In[ ]:


data.drop('two', axis=1)


# In[ ]:


data.drop(['two', 'four'], axis='columns')


# Como hemos dicho, por defecto, `drop` devuelve un nuevo objeto. Pero como otras funciones, podrían actuar de manera destructiva, **modificando el objeto original**. Para ello, hay que indicarlo con el argumento clave `inplace`:

# In[ ]:


data.drop('c', inplace=True)
data


# ### Indexado, selección y filtrado

# El acceso a los elementos de un objeto *Series* se hace de manera similar a los arrays de numpy, excepto que también podemos usar el correspondiente valor del índice, además de la posición numérica. Veámoslo con un ejemplo: 

# In[ ]:


s = pd.Series(np.arange(4.), index=['a', 'b', 'c', 'd'])
s


# Podemos acceder al segundo elemento de la serie anterir, bien mediante el valor `'b'`, o por la posición 1, ambos accesos son equivalentes:

# In[ ]:


s['b'], s[1]


# Más ejemplos de indexado en objetos de tipo *Series*:

# In[ ]:


s[2:4]


# In[ ]:


s[['b', 'a', 'd']]


# In[ ]:


s[[1, 3]]


# In[ ]:


s[s < 2]


# Podemos hacer también *slicing* con las etiquetas de un índice. Existe una diferencia importante, y es que el límite superior se considera incluido:

# In[ ]:


s['b':'c']


# Podemos incluso hacer **asignaciones** usando *slicing*, como en los arrays de numpy:

# In[ ]:


s['b':'c'] = 5
s


# Para *DataFrames*, el acceso mediante una etiqueta, extrae por defecto la correspondiente columna en forma de Series, como ya habíamos visto anteriormente. En el siguiente ejemplo: 

# In[ ]:


data = pd.DataFrame(
    np.arange(16).reshape((4, 4)),
    index=['Ohio', 'Colorado', 'Utah', 'New York'],
    columns=['one', 'two', 'three', 'four']
)
data


# In[ ]:


data['two'] ### 


# También se admite indexado mediante una lista de etiquetas:

# In[ ]:


data[['three', 'one']]


# Hay un par de casos particulares, que no funciona seleccionando columnas: si hacemos slicing con enteros, nos estamos refiriendo a las filas:

# In[ ]:


data[:2]


# También el indexado booleano filtar por filas:

# In[ ]:


data[data['three'] > 5]


# ### Selección con loc e iloc

# Además de los métodos directos de indexado que acabamos de ver, existen otros dos métodos para seleccionar datos en pandas

# - `loc`: es manera de acceder a los datos de un *DataFrame* usando las etiquetas de las filas y columnas. También se utiliza para indexar con booleanos.
# 
# - `iloc`: podemos usar índices enteros, como con numpy. 
# 
# Veamos algunos ejemplos.

# In[ ]:


data


# Para acceder a la fila etiquetada como `Colorado` y sólo a las columnas `two` y `three`, en ese orden (nótese que se devuelve una serie):

# In[ ]:


data.loc['Colorado', ['two', 'three']]


# Un ejemplo, similar, pero ahora con índices numéricos. La fila de índice 2, sólo con las columnas 3, 0 y 1. 

# In[ ]:


data.iloc[2, [3, 0, 1]]


# La fila de índice 2:

# In[ ]:


data.iloc[2]


# Podemos especificar una subtabla por sus filas y columnas

# In[ ]:


data.iloc[[1, 2], [3, 0, 1]]


# Podemos usar slicing con las etiquetas (recordar que el límite superior es inclusive):

# In[ ]:


data.loc[:'Utah', 'two']


# Un ejemplo algo más complicado. Seleccionamos primero las tres primeras columnas mediante slicing con enteros, y luego seleccionamos las filas que en la columna etiquetada con `'three'` tienen un valor mayor que 5:

# In[ ]:


data.iloc[:, :3][data.three > 5]


# :::{exercise}
# :label: pandas-loc-iloc
# 
# Carga el siguiente dataframe
# ```
# url = "https://github.com/mattharrison/datasets/raw/master/data/vehicles.csv.zip"
# df = pd.read_csv(url)
# ```
# - Utiliza el método `set_index` para hacer que la columna `make` se convierta en el índice
# - Devuelve las primera 3 filas que corresponden a `make == 'Ferrari'`.
# - Devuelve las 5 primeras columnas de aquellas filas que tengan `city08` mayor que 50
# 
# :::

# ### Operaciones aritméticas con valores de relleno

# In[ ]:


df1 = pd.DataFrame(
    np.arange(12.).reshape((3, 4)),
    columns=list('abcd')
)
df2 = pd.DataFrame(
    np.arange(20.).reshape((4, 5)),
    columns=list('abcde')
)

df1.loc[1, 'b'] = np.nan
df2.loc[1, 'b'] = np.nan


# In[ ]:


df1


# In[ ]:


df2


# Al sumar ambos *DataFrames*,téngase en cuanta que la fila 3 no existe en una de ellas, al igual que la columna `'e'`. POr tanto, en el resultado, se crearán en esas fila y columna con valores no determinados: 

# In[ ]:


df1 + df2


# Sin embargo, podemos usar el método `add` con "valor de relleno" 0, y en ese caso, cuando uno de los operandos no esté definido, se tome ese valor por defecto (cero en este caso). Nótese que si ninguno de los operandos está definido (como en `(1,'b')`), entonces no se aplica el relleno. 

# In[ ]:


df1.add(df2, fill_value=0)


# Como `add`, existe otras operaciones aritméticas que permiten `fill_value`: `sub`, `mul`, `div`, `pow`, ...

# Relacionado con esto, también es interesante destacar que cuando se reindexa un objeto, podemos especificar el valor de relleno, cuando el valor no esté especificado en el objeto original:

# In[ ]:


df1.reindex(columns=df2.columns, fill_value=0)


# ### Aplicación de funciones a las "vectorizadas"

# Como en numpy, podemos aplicar funciones de forma vectorizada a todos los valores de un *Series* o un *DataFrame*:

# In[ ]:


frame = pd.DataFrame(
    np.random.randn(4, 3), 
    columns=list('bde'),
    index=['Utah', 'Ohio', 'Texas', 'Oregon']
)
frame


# Por ejemplo, aplicamos valor absoluto a cada elemento de la tabla:

# In[ ]:


np.abs(frame)


# Otra operación muy frecuente consiste en aplicar una función definida sobre arrays unidimensionales, a lo largo de uno de los ejes (filas o columnas). Esto se hace con el método `apply` de los *DataFrames*.

# Supongamos que queremos calcular la diferencia entre el máximo y el mínimo de cada columna de una tabla. Lo podemos hacer así:

# In[ ]:


f = lambda x: x.max() - x.min()
frame.apply(f)


# El eje por defecto para hacer un `apply` es el 0, es decir el de las filas (y por tanto aplica la opración sobre cada columna). Podemos usar el argumento `axis` para especificar que queremos aplicar en el sentido de las columnas (y por tanto, hacer el cálculo sobre las filas): 

# In[ ]:


frame.apply(f, axis='columns')


# En realidad, hay muchas funciones (como `sum` o `mean`) que de por sí ya están adaptadas a *DataFrames* y no necesitan usar apply, como veremos más adelante. 

# Compara las siguientes ejecuciones usando `apply` recorriendo filas o una función vectorizada

# In[ ]:


url = "https://github.com/mattharrison/datasets/raw/master/data/vehicles.csv.zip"
df = pd.read_csv(url)


# In[ ]:


def gt20(val):
    return val > 20


# In[ ]:


get_ipython().run_cell_magic('timeit', '', 'df.city08.apply(gt20)\n')


# In[ ]:


get_ipython().run_cell_magic('timeit', '', 'df.city08.gt(20)\n')


# :::{exercise}
# :label: pandas-vfunctions
# 
# Selecciona las columnas numéricas de `df` con el método `select_dtypes` y normaliza las columnas.
# 
# :::

# ## Ordenación

# En pandas tenemos posibilidad de ordenar bien teniendo en cuenta las etiquetas de las filas o columnas, o bien con los valores propiamente dichos:

# Comenzamos con un ejemplo con *Series*:

# In[ ]:


s = pd.Series(range(4), index=['d', 'a', 'b', 'c'])
s


# Con `sort_index` ordenamos la serie por las etiquetas del índice:

# In[ ]:


s.sort_index()


# Con *DataFrame*, podemos ordenar por las etiquetas de las filas, o también por las columnas, usando el argumento `axis`:

# In[ ]:


frame = pd.DataFrame(
    np.arange(8).reshape((2, 4)),
    index=['three', 'one'],
    columns=['d', 'a', 'b', 'c']
)
frame


# In[ ]:


frame.sort_index()


# In[ ]:


frame.sort_index(axis=1)


# In[ ]:


frame.sort_index(axis=1, ascending=False) # se puede especificar si es ascendente o descendente


# Con `sort_values`, ordenamos por los valores de las entradas. Por ejemplo, en una serie:

# In[ ]:


s = pd.Series([4, 7, -3, 2])
s.sort_values()


# Si ordenamos por valores, los *NaN* se sitúan al final:

# In[ ]:


s = pd.Series([4, np.nan, 7, np.nan, -3, 2])
s.sort_values()


# Con `sort_values` aplicado a *DataFrames*, podemos ordenar por el valor de alguna columna, o incluso por el valor de una fila, usando el argumento clave `'by'`:

# In[ ]:


frame = pd.DataFrame(
    {
        'b': [4, 7, -3, 2], 
        'a': [0, 1, 0, 1]
    },
    index=['j','k','l','m']
)
frame


# Ordenación de la tabla según la columna `'b'`:

# In[ ]:


frame.sort_values(by='b')


# Por el valor de dos columnas (lexicográficamente):

# In[ ]:


frame.sort_values(by=['a', 'b'])


# Por el valor de una fila:

# In[ ]:


frame.sort_values(axis=1, by='k')


# ## Funciones estadísticas descriptivas

# Los objetos de pandas incorporan una serie de métodos estadísticos que calculan un valor a partir de los valores de una serie o de filas o columnas de un *DataFrame*. Una particularidad interesante es que manejan adecuadamente los valores no especificados. Veamos algunos ejemplos:

# In[ ]:


frame = pd.DataFrame(
    [
        [1.4, np.nan], 
        [7.1, -4.5], 
        [np.nan, np.nan], 
        [0.75, -1.3]
    ],
    index=['a', 'b', 'c', 'd'],
    columns=['one', 'two']
)
frame


# Por defecto, el método `sum` calcula la suma de cada columna de un *DataFrame*. Los valores NaN se tratan como 0 (a no ser que toda la serie sea de valores NaN):

# In[ ]:


frame.sum()


# Como es habitual, con el parámetro `axis` podemos hacerlo por el eje de las columnas:

# In[ ]:


frame.sum(axis='columns')


# Con `mean` calculamos la media de filas o columnas según el eje elegido con *axis*. El parámetro `skipna` nos permite indicar si se excluyen o no los valores NaN:

# In[ ]:


frame.mean(axis='columns', skipna=False)


# El método `idxmax` nos da la etiqueta donde se alcanza el mínimo de cada columna (o cada fila)

# In[ ]:


frame.idxmax()


# El método `cumsum` nos da los acumulados por fila o por columna:

# In[ ]:


frame.cumsum()


# Por último el método `describe` produce un resumen con las estadísticas más importantes:

# In[ ]:


frame.describe()


# Para tipos no numéricos, `describe` también devuelve información

# In[ ]:


s = pd.Series(['a', 'a', 'b', 'c'] * 4)
s.describe()


# :::{exercise}
# :label: pandas-housing
# 
# Descarga el dataframe `housing` utilizando el siguiente código
# 
# ```
# import os
# import tarfile
# import urllib
# import pandas as pd
# 
# DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
# HOUSING_PATH = os.path.join("data", "housing")
# HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
# 
# def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
#     os.makedirs(housing_path, exist_ok=True)
#     tgz_path = os.path.join(housing_path, "housing.tgz")
#     urllib.request.urlretrieve(housing_url, tgz_path)
#     housing_tgz = tarfile.open(tgz_path)
#     housing_tgz.extractall(path=housing_path)
#     housing_tgz.close()
# 
# def load_housing_data(housing_path=HOUSING_PATH):
#     csv_path = os.path.join(housing_path, "housing.csv")
#     return pd.read_csv(csv_path)
# 
# fetch_housing_data()
# housing = load_housing_data()
# ```
# 
# Realiza un análisis rápido de las variables númericas y categóricas. Rellena los valores faltantes con `fillna` y realiza las visualizaciones que veas adecuadas.
# 
# :::
