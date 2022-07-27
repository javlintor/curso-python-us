#!/usr/bin/env python
# coding: utf-8

# # Tema 1: Herramientas python para ciencia de los datos

# ## 1.2 Introducción a pandas
# 
# Los ejemplos y la presentación que sigue está tomada del capítulo 4 del libro:
# 
# [*Python for Data Analysis*](https://wesmckinney.com/pages/book.html)  
# **Wes McKinney**  
# O'Reilly 2017
# 
# Github con el material del libro: [Github](https://github.com/wesm/pydata-book)

# In[1]:


import pandas as pd


# In[2]:


#from pandas import Series, DataFrame


# In[3]:


import numpy as np
np.random.seed(12345)
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(10, 6))
PREVIOUS_MAX_ROWS = pd.options.display.max_rows
pd.options.display.max_rows = 20
np.set_printoptions(precision=4, suppress=True)


# ### *Series* de pandas: introducción

# Una *Serie* en pandas es una estructura tipo array unidimensional, junto con una secuencia (su *índice*) de "etiquetaa" que "nombran" a los elementos del array. El caso más simple: 

# In[4]:


obj = pd.Series([4, 7, -5, 3])
obj


# En el ejemplo, el índice es la columna de la izquierda (por defecto se ha creado un índice numérico) y los valores de la serie son la columna de la derecha.
# 
# Con los atributos `values` e `index` accedemos al array de valores y al índice:

# In[5]:


obj.values


# In[6]:


obj.index  # es similar a range(4), pero de una clase `RangeIndex`, específica pandas


# Podemos crear una serie proporcionando explícitamente el índice:

# In[7]:


obj2 = pd.Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
obj2


# In[8]:


obj2.index


# El índice en una serie de pandas, generaliza la nocion de índice en numpy. Podemos acceder y modificar los valores de una serie a través del índice, 

# In[9]:


obj2['a']


# In[10]:


obj2['d'] = 6
obj2


# Podemos incluso obtener una subserie, mediante un subconjunto reordenado de sus índices:

# In[11]:


obj2[['c', 'a', 'd']]


# También muchas de las funcionalidades de los arrays de numpy, como por ejemplo el indexado booleano o las operaciones vectorizadas, se aplican también a las series:

# In[12]:


obj2[obj2 > 0]


# In[13]:


obj2 * 2


# In[14]:


np.exp(obj2)


# Una manera muy útil de pensar en una serie de pandas es como un diccionario (en el que el índice son las claves) de longitud fija y en el que el orden es significativo. Por ejemplo, para consultar si una etiqueta está en un índice de una serie, podemos usar `in`, como en los diccionarios:

# In[15]:


'b' in obj2


# In[16]:


'e' in obj2


# De hecho, una manera muy frecuente de crear una serie es a partir de un diccionario. Las claves se ordenarán y formarán el índice de la serie, como en el siguiente ejemplo:

# In[17]:


sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj3 = pd.Series(sdata)
obj3


# Si queremos introducir un orden específico entre las claves del diccionario, entonces podemos combinar el pasar el diccionario junto con la lista de etiquetas ordenadas:

# In[18]:


states = ['California', 'Ohio', 'Oregon', 'Texas']
obj4 = pd.Series(sdata, index=states)
obj4


# Nótese que solo se incluye en el índice lo incluido en la lista (por ejemplo `Utah` no forma parte del índice a pesar de que es una clave del diccionario). Como `California` no es una clave del diccionario, pero se ha incluido en el índice, se incluye con valor `NaN`(*Not a Number*), que es la manera en pandas para indicar valores inexistentes. 

# Con `isnull` podemos localizar qué entradas de la serie tienen valores inexistentes:

# In[19]:


pd.isnull(obj4)


# In[20]:


obj4.isnull()


# En las series, como con los arrays de numpy, podemos realizar operaciones vectorizadas. Lo interesante aquí es que las operaciones se *alinean* por las correspondientes etiquetas. Por ejemplo:

# In[21]:


obj3


# In[22]:


obj4


# In[23]:


obj3 + obj4


# Como se observa, se han sumado los correspondientes valores en aquellas entradas en los que el valor estaba disponible. En el caso de California, la operación no se ha podido realizar ya que en una de las series no estaba (y además en la otra estaba con valor `NaN`). En el caso de Utah, sólo estaba en una de las series.

# Tanto la serie como el índice de la serie tienen un atributo nombre, del cual veremos más adelante su utilidad:

# In[24]:


obj4.name = 'population'
obj4.index.name = 'state'
obj4


# In[25]:


obj


# Por último, podemos modificar (destructivamente) un índice mediante la correspondiente asignación al atributo `index`:

# In[26]:


obj.index = ['Bob', 'Steve', 'Jeff', 'Ryan']
obj


# ### *DataFrame* de pandas: introducción

# Un *DataFrame* de pandas es una tabla bidimensional, con las columnas y las filas en un determinado orden. Cada columna puede ser de un tipo diferente. En términos de índices: tanto las filas como las columnas están indexadas. Puede ser visto como un diccionario en el que las claves son las etiquetas de las columnas, y todos los valores son *Series* de pandas que comparten el mismo índice. 
# 
# Aunque hay muchas maneras de crear un *DataFrame*, una de las más frecuentes es mediante un diccionario cuyos valores asociados a las claves son listas de la misma longitud. Por ejemplo:   

# In[27]:


data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002, 2003],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
frame = pd.DataFrame(data)


# Nótese la forma en que se muestra un *DataFrame* en Jupyter:

# In[28]:


frame


# Cuando los *DataFrames* son grandes, puede ser útil el método `heads`, que muestar solo las primeras cinco filas de la tabla:

# In[29]:


frame.head()


# Como en el caso de las Series, podemos proporcionar las columnas en un orden determinado: 

# In[30]:


pd.DataFrame(data, columns=['year', 'state', 'pop'])


# Y también podemos indicar expresamente el índice de las filas:

# In[31]:


pd.DataFrame(data, columns=['pop', 'year', 'state'], index=['one', 'two', 'three', 'four',
                             'five', 'six'])


# Si al proporcionar los nombres de las columnas damos una de ellas que no aparece en el diccionario con los datos, entonces se crea la columnos con valores no determinados:

# In[32]:


frame2 = pd.DataFrame(data, columns=['year', 'state', 'pop', 'debt'],
                      index=['one', 'two', 'three', 'four',
                             'five', 'six'])
frame2


# Los atributos `index` y `columns` nos devuelven los correspondientes índices de las filas y de las columnas (ambos son objetos `Index` de pandas): 

# In[33]:


frame2.index


# In[34]:


frame2.columns


# Para acceder a una columna en concreto del *DataFrame*, podemos hacerlo usando la notación de diccionario, o también como atributo. En ambos casos se devuelve la correspondiente columna como un objeto *Series* de pandas: 

# In[35]:


frame2['state']


# In[36]:


frame2.year


# De igual manera, podemos acceder a una fila del *DataFrame* mediante el atributo `loc`. La fila también se devuelve como un objeto *Series*, cuyo índice está formado por los nombres de las columnas:

# In[37]:


frame2.loc['three']


# Veamos ahora ejemplos sobre cómo podemos modificar columnas mediante asignaciones. En general, muchas de los procedimientos de numpy aquí también son válidos, pero teniendo en cuenta que indexamos mediante el nombre de la columna: 

# Por ejemplo, asignar el mismo valor a toda una columna:

# In[38]:


frame2['debt'] = 16.5
frame2


# O asignar mediante una secuencia:

# In[39]:


frame2['debt'] = np.arange(6.)
frame2


# Cuando a una columna le asignamos una lista o un array, como en el ejemplo anterior, la longitud de la secuencia debe de coincidir con el número de filas del *DataFrame*. Sin embargo, podemos asignar con un objeto *Series* y los valores se asignarán alineando por el valor del índice, incluso parcialmente (al resto se el asignará *NaN*):

# In[40]:


val = pd.Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])
frame2['debt'] = val
frame2


# Si asignamos una columna que no existe, ésta se creará. Por ejemplo:

# In[41]:


frame2['eastern'] = frame2.state == 'Ohio'
frame2


# Con `del` borranos una columna completa:

# In[42]:


del frame2['eastern']
frame2.columns


# In[43]:


frame2


# Otra forma de crear un *DataFrame* es a partir de un diccionario de diccionarios, en el que las claves externas constituyen las etiquetas de las columnas, y las internas como las de las filas:

# In[44]:


pop = {'Nevada': {2001: 2.4, 2002: 2.9},
       'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}


# In[45]:


frame3 = pd.DataFrame(pop)
frame3


# Como en numpy, podemos también obtener la traspuesta de un *DataFrame*, quedando las filas como columns y viceversa:

# In[46]:


frame3.T


# Si usamos un diccionario anidado, y le damos explícitamente el índice como una secuencia, entonces las filas se tienen en cuenta y se ordenan como se indica en la secuencia. Lo mismo ocurre si se le especifican explícitamente las columnas:

# In[47]:


pd.DataFrame(pop, index=[2001, 2002, 2003],columns=["Ohio","California","Nevada"])


# También se puede dar un *DataFrame* como un diccionario en el que cada clave (columna) tiene asociada una serie: 

# In[48]:


pdata = {'Ohio': frame3['Ohio'][:-1],
         'Nevada': frame3['Nevada'][:2]}
pd.DataFrame(pdata)


# Con el atributo `name` (tanto de `index`como de `columns`) podemos acceder y/o modificar el nombre de las filas y las columnas, que se mostrarán al mostrarse la tabla:

# In[49]:


frame3.index.name = 'year'; frame3.columns.name = 'state'
frame3


# Por último, mediante `values`, accedemos a un array bidimensional con los valores de cada entrada de la tabla:

# In[50]:


frame3.values


# In[51]:


frame2.values # el dtype se acomoda a lo más general. 


# #### Resumen de algunas maneras de crear un *DataFrame*:
# 
# * Array bidimensional, opcionalmente con `index`y/o `column`
# * Diccionario de arrays, listas o tuplas de la misma longitud; cada clave se refiere a una columna
# * Diccionario de *Series*; cada clave es una columna y las filas se alinean según los índices de las series, o bien se le pasa explícitamente el índice. 
# * Diccionario de diccionarios: las claves externas son las columnas, las internas las filas.
# * Lista de listas o tuplas: como en el caso de array bidimensional. 

# ### Funcionalidades básicas

# #### Reindexado

# Mediante reindexado, creamos un nuevo objeto adaptándolo a un nuevo índice. Veámos lo con un ejemplo:

# In[52]:


obj = pd.Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])
obj


# Reindexamos con `reindex`. Nótese cómo la nueva etiqueta `'e'` se incluye en la nueva serie, con valor no especificado: 

# In[53]:


obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e'])
obj2


# Se puede reindexar un *DataFrame*, ajustando tanto las filas como las columnas. Veámoslo con un ejemplo:

# In[54]:


frame = pd.DataFrame(np.arange(9).reshape((3, 3)),
                     index=['a', 'c', 'd'],
                     columns=['Ohio', 'Texas', 'California'])
frame


# Si a `reindex`se le pasa solamente una secuencia, entonces se entiende que se quieren reajustar las filas:

# In[55]:


frame2 = frame.reindex(['a', 'b', 'c', 'd'])
frame2


# Para reindexar las columnas, hay que pasar el argumento clave `columns`:

# In[56]:


states = ['Texas', 'Utah', 'California']
frame.reindex(columns=states)


# #### Eliminando entradas de un eje

# Ya vimos antes que `del` borra una columna completa de un *DataFrame*. Mediante `drop`, podemos *crear nuevos objetos* resultantes de eliminar filas o columnas completas. Veamos algunos ejemplos. En primer lugar, con las *Series*:

# In[57]:


obj = pd.Series(np.arange(5.), index=['a', 'b', 'c', 'd', 'e'])
obj


# In[58]:


new_obj = obj.drop('c')
new_obj


# In[59]:


obj.drop(['d', 'c'])


# Ahora veamos el uso de `drop` con *DataFrames*:

# In[60]:


data = pd.DataFrame(np.arange(16).reshape((4, 4)),
                    index=['Ohio', 'Colorado', 'Utah', 'New York'],
                    columns=['one', 'two', 'three', 'four'])
data


# Por defecto, se eliminan del eje 0 (las filas):

# In[61]:


data.drop(['Colorado', 'Ohio'])


# Podemos eliminar columnas, indicándo que se quiere hacer en `axis=1` o `axis='columns'`:

# In[62]:


data.drop('two', axis=1)


# In[63]:


data.drop(['two', 'four'], axis='columns')


# Como hemos dicho, por defecto, `drop` devuelve un nuevo objeto. Pero como otras funciones, podrían actuar de manera destructiva, modificando el objeto original. Para ello, hay que indicarlo con el argumento clave `inplace`:

# In[64]:


obj.drop('c', inplace=True)
obj


# ### Indexado, selección y filtrado

# El acceso a los elementos de un objeto *Series* se hace de manera similar a los arrays de numpy, excepto que también podemos usar el correspondiente valor del índice, a demás de la posición numérica. Veámoslo con un ejemplo: 

# In[65]:


obj = pd.Series(np.arange(4.), index=['a', 'b', 'c', 'd'])
obj


# Podemos acceder al segundo elemento de la serie anterir, bien mediante el valor `'b'`, o por la posición 1, ambos accesos son equivalentes:

# In[66]:


obj['b'],obj[1]


# Más ejemplos de indexado en objetos de tipo *Series*:

# In[67]:


obj[2:4]


# In[68]:


obj[['b', 'a', 'd']]


# In[69]:


obj[[1, 3]]


# In[70]:


obj[obj < 2]


# Podemos hacer también *slicing* con las etiquetas de un índice. Existe una diferencia importante, y es que el límite superior se considera incluido:

# In[71]:


obj['b':'c']


# Podemos incluso establecer valores usando *slicing*, como en los arrays de numpy:

# In[72]:


obj['b':'c'] = 5
obj


# Para *DataFrames*, el acceso mediante una etiqueta, extrae por defecto la correspondiente columna en forma de Series, como ya habíamos visto anteriormente. En el siguiente ejemplo: 

# In[73]:


data = pd.DataFrame(np.arange(16).reshape((4, 4)),
                    index=['Ohio', 'Colorado', 'Utah', 'New York'],
                    columns=['one', 'two', 'three', 'four'])
data


# In[74]:


data['two'] ### 


# También se admite indexado mediante una lista de etiquetas:

# In[75]:


data[['three', 'one']]


# Hay un par de casos particulares, que no funciona seleccionando columnas: si hacemos slicing con enteros, nos estamos refiriendo a las filas:

# In[76]:


data[:2]


# También el indexado booleano filtar por filas:

# In[77]:


data[data['three'] > 5]


# #### Selección con loc e iloc

# Con `loc` tenemos una manera de acceder a los datos de un *DataFrame* de una manera similar a los array de numpy, usando las etiquetas de las filas y columnas. Con `iloc` podemos usar índices enteros, como con numpy. Veamos algunos ejemplos.

# In[78]:


data


# Para acceder a la fila etiquetada como `Colorado` y sólo a las columnas `two` y `three`, en ese orden (nótese que se devuelve una serie):

# In[79]:


data.loc['Colorado', ['two', 'three']]


# Un ejemplo, similar, pero ahora con índices numéricos. La fila de índice 2, sólo con las columnas 3, 0 y 1. 

# In[80]:


data.iloc[2, [3, 0, 1]]


# La fila de índice 2:

# In[81]:


data.iloc[2]


# Podemos especificar una subtabla por sus filas y columnas

# In[82]:


data.iloc[[1, 2], [3, 0, 1]]


# Podemos usar slicing con las etiquetas (recordar que el límite superior es inclusive):

# In[83]:


data.loc[:'Utah', 'two']


# Un ejemplo algo más complicado. Seleccionamos primero las tres primeras columnas mediante slicing con enteros, y luego seleccionamos las filas que en la columna etiquetada con `'three'` tienen un valor mayor que 5:

# In[84]:


data.iloc[:, :3][data.three > 5]


# ### Operaciones aritméticas con alineamiento de índices

# Aunque ya lo hemos visto anteriormente, hagamos de nuevo hincapié en que al hacer operaciones aritméticas entre objetos, estas se hacen alineando las correspondientes eetiquetas de los índices (y dejando NaN cuando alguno de los operandos no está definido):

# Veamos un ejemplo con *Series*:

# In[85]:


s1 = pd.Series([7.3, -2.5, 3.4, 1.5], index=['a', 'c', 'd', 'e'])
s2 = pd.Series([-2.1, 3.6, -1.5, 4, 3.1],
               index=['a', 'c', 'e', 'f', 'g'])


# In[86]:


s1


# In[87]:


s2


# In[88]:


s1 + s2


# En el caso de los *DataFrames*, el alineamiento se produce en las filas y columnas:

# In[89]:


df1 = pd.DataFrame(np.arange(9.).reshape((3, 3)), columns=list('bcd'),
                   index=['Ohio', 'Texas', 'Colorado'])
df2 = pd.DataFrame(np.arange(12.).reshape((4, 3)), columns=list('bde'),
                   index=['Utah', 'Ohio', 'Texas', 'Oregon'])


# In[90]:


df1


# In[91]:


df2


# In[92]:


df1 + df2


# #### Operaciones aritméticas con valores de relleno

# In[93]:


df1 = pd.DataFrame(np.arange(12.).reshape((3, 4)),
                   columns=list('abcd'))
df2 = pd.DataFrame(np.arange(20.).reshape((4, 5)),
                   columns=list('abcde'))

df1.loc[1, 'b'] = np.nan
df2.loc[1, 'b'] = np.nan


# In[94]:


df1


# In[95]:


df2


# Al sumar ambos *DataFrames*,téngase en cuanta que la fila 3 no existe en una de ellas, al igual que la columna `'e'`. POr tanto, en el resultado, se crearán en esas fila y columna con valores no determinados: 

# In[96]:


df1 + df2


# Sin embargo, podemos usar el método `add` con "valor de relleno" 0, y en ese caso, cuando uno de los operandos no esté definido, se tome ese valor por defecto (cero en este caso). Nótese que si ninguno de los operandos está definido (como en `(1,'b')`), entonces no se aplica el relleno. 

# In[97]:


df1.add(df2, fill_value=0)


# Como `add`, existem otras operaciones aritméticas que permiten `fill_value`: `sub`, `mul`, `div`, `pow`, ...

# Relacionado con esto, también es interesante destacar que cuando se reindexa un objeto, podemos especificar el valor de relleno, cuando el valor no esté especificado en el objeto original:

# In[98]:


df1.reindex(columns=df2.columns, fill_value=0)


# ### Aplicación de funciones a las "vectorizadas"

# Como en numpy, podemos aplicar funciones de forma vectorizada a todos los valores de un *Series* o un *DataFrame*:

# In[99]:


frame = pd.DataFrame(np.random.randn(4, 3), columns=list('bde'),
                     index=['Utah', 'Ohio', 'Texas', 'Oregon'])
frame


# Por ejemplo, aplicamos valor absoluto a cada elemento de la tabla:

# In[100]:


np.abs(frame)


# Otra operación muy frecuente consiste en aplicar una función definida sobre arrays unidimensionales, a lo largo de uno de los ejes (filas o columnas). Esto se hace con el método `apply`de los *DataFrames*.
Supongamos que queremos calcular la diferencia entre el máximo y el mínimo de cada columna de una tabla. Lo podemos hacer así:
# In[101]:


f = lambda x: x.max() - x.min()
frame.apply(f)


# El eje por defecto para hacer un `apply`es el 0, es decir el de las filas (y por tanto aplica la opración sobre cada columna). Podemos usar el argumento `axis` para especificar que queremos aplicar en el sentido de las columnas (y por tanto, hacer el cálculo sobre las filas): 

# In[102]:


frame.apply(f, axis='columns')


# En realidad, hay muchos funciones estadísticas (como `sum` o `mean`) que de por sí ya están adaptadas a *DataFrames* y no necesitan usar apply, como veremos más adelante. 

# ### Ordenación

# En pandas tenemos posibilidad de ordenar bien teniendo en cuenta las etiquetas de las filas o columnas, o bien con los valores propiamente dichos:

# Comenzamos con un ejemplo con *Series*:

# In[103]:


obj = pd.Series(range(4), index=['d', 'a', 'b', 'c'])
obj


# Con `sort_index` ordenamos la serie por las etiquetas del índice:

# In[104]:


obj.sort_index()


# Con *DataFrame*, podemos ordenar por las etiquetas de las filas, o también por las columnas, usando el argumento `axis`:

# In[105]:


frame = pd.DataFrame(np.arange(8).reshape((2, 4)),
                     index=['three', 'one'],
                     columns=['d', 'a', 'b', 'c'])
frame


# In[106]:


frame.sort_index()


# In[107]:


frame.sort_index(axis=1)


# In[108]:


frame.sort_index(axis=1, ascending=False) # se puede especificar si es ascendente o descendente


# Con `sort_values`, ordenamos por los valores de las entradas. Por ejemplo, en una serie:

# In[109]:


obj = pd.Series([4, 7, -3, 2])
obj.sort_values()


# Si ordenamos por valores, los *NaN* se sitúan al final:

# In[110]:


obj = pd.Series([4, np.nan, 7, np.nan, -3, 2])
obj.sort_values()


# Con `sort_values` aplicado a *DataFrames*, podemos ordenar por el valor de alguna columna, o incluso por el valor de una fila, usando el argumento clave `'by'`:

# In[111]:


frame = pd.DataFrame({'b': [4, 7, -3, 2], 'a': [0, 1, 0, 1]},index=['j','k','l','m'])
frame


# Ordenación de la tabla según la columna `'b'`:

# In[112]:


frame.sort_values(by='b')


# Por el valor de dos columnas (lexicográficamente):

# In[113]:


frame.sort_values(by=['a', 'b'])


# Por el valor de una fila:

# In[114]:


frame.sort_values(axis=1,by='k')


# ### Funciones estadísticas descriptivas

# Los objetos de pandas incorporan una serie de métodos estadísticos que calculan un valor a partir de los valores de una serie o de filas o columnas de un *DataFrame*. Una particularidad interesante es que manejan adecuadamente los valores no especificados. Veamos algunos ejemplos:

# In[115]:


df = pd.DataFrame([[1.4, np.nan], [7.1, -4.5],
                   [np.nan, np.nan], [0.75, -1.3]],
                  index=['a', 'b', 'c', 'd'],
                  columns=['one', 'two'])
df


# Por defecto, el método `sum` calcula la suma de cada columna de un *DataFrame*. Los valores NaN se tratan como 0 (a no ser que toda la serie sea de valores NaN):

# In[116]:


df.sum()


# Como es habitual, con el parámetro `axis` podemos hacerlo por el eje de las columnas:

# In[117]:


df.sum(axis='columns')


# Con `mean` calculamos la media de filas o columnas según el eje elegido con *axis*. El parámetro `skipna` nos permite indicar si se excluyen o no los valores NaN:

# In[118]:


df.mean(axis='columns', skipna=False)


# El método `idxmax`nos da la etiqueta donde se alcanza el mínimo de cada columna (o cada fila)

# In[119]:


df.idxmax()


# El método `cumsum` nos da los acumulados por fila o por columna:

# In[120]:


df.cumsum()


# Por último el método `describe` produce un resumen con las estadísticas más importantes:

# In[121]:


df.describe()


# In[122]:


obj = pd.Series(['a', 'a', 'b', 'c'] * 4)
obj.describe()

