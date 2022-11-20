#!/usr/bin/env python
# coding: utf-8

# (numpy)=
# # Introducción a Numpy

# Numpy es quizás el paquete de **computación numérica** más importante de Python. Se desarrolló como un paquete completo de álgebra lineal de código abierto para Python que podía rivalizar con MATLAB y similares. Es una biblioteca de Python con una larga historia y mucha funcionalidad, ya sea directamente en ella o construida a su alrededor (ver [SciPy](https://scipy.org/) y diferentes scikits). Es la base de otros paquetes del ecosistema de ciencia de datos 
# - Extraer, transformar y cargar datos: [Pandas](https://pandas.pydata.org/), [Dask](https://www.dask.org/), [OpenCV](https://opencv.org/)
# - Visualización de datos: [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/).
# - Modelos y evalución: [Scikit-learn](https://scikit-learn.org/stable/), [statsmodels](https://www.statsmodels.org/stable/index.html), [spaCy](https://spacy.io/)
# - Repote: Dash, [Stramlit](https://streamlit.io/)
# 
# La clave es que implementa arrays (matrices) multidimensionales de manera muy eficiente, con numerosas funcionalidades optimizadas sobre dicha estructura de datos. 
# 
# > Es muy común en la comunidad python usar el alias `np` cundo importamos Numpy:

# In[34]:


import numpy as np


# Aunque en principio las listas de python podrían servir para representar array de varias dimensiones, la eficiencia de numpy es mucho mejor, al estar construido sobre una biblioteca de rutinas en lenguaje C. Además muchas de las operaciones numpy que actúan sobre todo el array, están optimizadas y permiten evitar los bucles `for`de python, que actúan más lentamente.
# 
# Lo que sigue es un ejemplo de un array de numpy unidimensional con un millón de componentes, y el análogo como lista python. 

# In[35]:


arr1 = np.arange(1000000)
list1 = list(range(1000000))


# Vamos a obtener el array resultante de multiplicar por 2 cada componente, y veamos el tiempo de CPU que se emplea. Nótese que en el caso de numpy, dicha operación se especifica simplemente como "multiplicar por 2" el array. En el caso de las listas, tenemos que usar un bucle `for` para la misma operación. Obsérvese la gran diferencia en el tiempo de ejecución:  

# In[37]:


get_ipython().run_line_magic('timeit', 'for _ in range(10): arr2 = arr1 * 2')
get_ipython().run_line_magic('timeit', 'for _ in range(10): list2 = [x * 2 for x in list1]')


# Comparamos en la siguiente gráfico el tiempo que tardamos en sumar los `n` primeros números usando listas de Python o arrays de numpy.

# In[38]:


import time
import matplotlib.pyplot as plt

get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")

def get_ex_time(f, tries=10):
    def wrapper(*args, **kwargs):
        ex_times = []
        for _ in range(tries):
            start = time.time()
            f(*args, **kwargs)
            end = time.time()
            ex_time = end - start 
            ex_times.append(ex_time)
        mean_ex_time = np.mean(ex_times)
        return mean_ex_time
    return wrapper

@get_ex_time
def get_duplicate_time_python(i):
    return list(2*x for x in range(i))

@get_ex_time
def get_duplicate_time_numpy(i):
    return 2*np.arange(i)

@get_ex_time
def get_sum_time_python(i):
    return sum(range(i))

@get_ex_time
def get_sum_time_numpy(i):
    return np.sum(np.arange(i))

n = [10**i for i in range(9)]
t_duplicate_python= [get_duplicate_time_python(i) for i in n]
t_duplicate_numpy = [get_duplicate_time_numpy(i) for i in n]
t_sum_python= [get_sum_time_python(i) for i in n]
t_sum_numpy = [get_sum_time_numpy(i) for i in n]


# In[41]:


fig, ax = plt.subplots(2, 1, figsize=(20, 15))
ax[0].plot(n, t_duplicate_python, "o-", label="Python")
ax[0].plot(n, t_duplicate_numpy, "o-", label="Numpy")
ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[0].set_xlabel("tamaño")
ax[0].set_ylabel("tiempo (s)")
ax[0].set_title("Multiplicar por 2 usando Python puro vs numpy")
ax[0].grid(True)
ax[0].legend()
ax[1].plot(n, t_sum_python, "o-", label="Python")
ax[1].plot(n, t_sum_numpy, "o-", label="Numpy")
ax[1].set_xscale("log")
ax[1].set_yscale("log")
ax[1].set_xlabel("n")
ax[1].set_ylabel("tiempo (s)")
ax[1].set_title("Sumar n primeros elementos usando Python puro vs numpy")
ax[1].grid(True)
ax[1].legend()
fig.show()


# ## Arrays de Numpy

# La estructura de datos principal de Numpy es el **array n-dimensional**. Como hemos dicho, Numpy nos permite operar sobre los arrays en su totalidad,  especificando las operaciones como si lo hiciéramos con las componentes individuales. 

# Hay muchas formas de crear arrays, pero vamos a empezar creando un array de números psuedoaletorios obtenidos como muestras de una distribución normal de media 0 y desviación típica 1 para empezar a explorar los atributos y posibles operaciones.

# In[ ]:


data = np.random.randn(2, 3)
data


# Podemos por ejemplo obtener el array resultante de multiplicar cada componente del array por 10, sin necesidad de hacerlo elemento a elemento

# In[ ]:


data * 10


# O la suma de cada componente consigo mismo:

# In[ ]:


data + data


# Nótese que las operaciones anteriores **no cambian el array sobre el que operan**:

# In[ ]:


data


# Los arrays de numpy deben ser **homogéneos**, es decir todas sus componentes del mismo tipo. Numpy incorpora una [una variedad extensa de tipos](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.byte), pero los más usuales son los de tipo `float`, `int` y `bool`.

# ![dtype-hierarchy.png](data:image/png;base64,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)

# In[ ]:


data.dtype


# El método `ndim` nos da el número de dimensiones, y el método `shape` nos da el tamaño de cada dimensión, en forma de tupla. 
# 
# En este caso, tenemos que `data` es un array bidimensional, con 2 "filas" y 3 "columnas":

# In[ ]:


print(data.ndim)
print(data.shape)


# En la terminología de numpy, cada una de las dimensiones se denominan "ejes" (*axis*), y se numeran consecutivamente desde 0. Por ejemplo, en un array bidimensional, el eje 0 (*axis=0*) corresponde a las filas y el eje 1 corresponde a las columnas (*axis=1*). 

# ### Creación de arrays

# La manera más fácil de crear arrays es mediante el método `np.array`. Basta con aplicarlo a cualquier objeto de tipo secuencial que sea susceptible de transformarse en un array. Por ejemplo, una lista de números se transforma en un array unidimensional: 

# In[ ]:


data1 = [6, 7.5, 8, 0, 1]
arr1 = np.array(data1)
arr1


# Si le pasamos listas anidadas con una estructura correcta, podemos obtener el correspondiente array bidimensional: 

# In[ ]:


data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
arr2 = np.array(data2)
arr2


# Hay otras maneras de crear arrays. Damos a continuación solo algunos ejemplos:

# In[ ]:


np.zeros(10)


# In[ ]:


np.ones((3, 6))


# In[ ]:


np.arange(20)


# In[ ]:


np.eye(7)


# In[ ]:


np.diag(range(5))


# In[ ]:


np.


# Podemos usar también `reshape` para transformar entre distintas dimensiones. Por ejemplo, un uso típico es crear un array bidimensional a partir de uno unidimensional:

# In[ ]:


L=np.arange(12)
M=L.reshape(3,4)
L,M


# Hasta ahora sólo hemos visto ejemplos de arrays unidimensionales y bidimensionales. Estas son las dimensiones más frecuentes, pero numpy soporta arrays con cualquier número finito de dimensiones. Por ejemplo, aquí vemos un array de tres dimensiones:

# In[ ]:


np.zeros((2, 3, 2))


# ### Tipos de datos en las componentes de un array (*dtype*)

# El tipo de dato de las componentes de un array está implícito cuando se crea, pero podemos especificarlo: 

# In[ ]:


arr1 = np.array([1, 2, 3], dtype=np.float64)
arr2 = np.array([1, 2, 3], dtype=np.int32)
print(arr1.dtype)
print(arr2.dtype)


# Podemos incluso convertir el `dtype` de un array que ya se ha creado, usando el método `astype`. En el ejemplo que sigue, a partir de un array de enteros, obtenemos uno de números de coma flotante:

# In[ ]:


arr = np.array([1, 2, 3, 4, 5])
arr.dtype


# In[ ]:


float_arr = arr.astype(np.float64)
float_arr.dtype
float_arr


# Podemos incluso pasar de coma flotante a enteros, en cuyo caso se trunca la parte decimal:

# In[ ]:


arr = np.array([3.7, -1.2, -2.6, 0.5, 12.9, 10.1])
arr
arr.astype(np.int32)


# O pasar de strings a punto flotante, siempre que los strings del array tengan sentido como números:

# In[ ]:


numeric_strings = np.array(['1.25', '-9.6', '42'], dtype=np.string_)
numeric_strings.astype(float)


# ### Operaciones aritméticas con arrays

# Una de las características más interesantes de numpy es la posibilidad de aplicar eficientes operaciones aritméticas "componente a componente" entre arrays, sin necesidad de usar bucles `for`. Basta con usar la correspondiente operación numérica de python. 
# 
# Veamos algunos ejemplos:

# In[ ]:


A = np.array([[1., 2., 3.], [4., 5., 6.]])
B = np.array([[8.1, -22, 12.3], [6.1, 7.8, 9.2]])


# In[ ]:


A * B


# In[ ]:


A - B


# En principio, para poder aplicar estas operaciones entre arrays, se deben aplicar sobre arrays con las mismas dimensiones (el mismo `shape`). Es posible operar entre arrays de distintas dimensiones, con el mecanismo de *broadcasting*, pero es ésta una característica avanzada de numpy que no veremos aquí.  

# Podemos efectuar operaciones entre arrays y números (*escalares*), indicando con ello que la operación con el escalar se aplica a cada uno de los componentes del array. Vemos algunos ejemplos: 

# In[ ]:


3+B


# In[ ]:


1 / A


# In[ ]:


A ** 0.5


# Igualmente, podemos efectuar comparaciones aritméticas entre dos arrays, obteniendo el correspondiente array de booleanos como resultado:

# In[ ]:


A > B-5


# En todos los casos anteriores, nótese que estas operaciones no modifican los arrays sobre los que se aplican, sino que obtienen un nuevo array con el correspondiente resultado.

# ### Indexado y *slicing* (operaciones básicas) 

# Otra de las características más interesantes de numpy es la gran flexibilidad para acceder a las componentes de un array, o a un subconjunto del mismo. Vamos a ver a continuación algunos ejemplos básicos.

# **Arrays unidimensonales**

# Para arrays unidimensionales, el acceso es muy parecido al de listas. Por ejemplo, acceso a las componentes:

# In[ ]:


C = np.arange(10)*2
C


# In[ ]:


C[5]


# La operación de *slicing* en arrays es similar a la de listas. Por ejemplo:

# In[ ]:


C[5:8]


# Sin embargo, hay una diferencia fundamental: en general en python, el slicing siempre crea *una copia* de la secuencia original. En numpy, el *slicing* es una *vista* de array original. Esto tiene como consecuencia que las modificaciones que se realicen sobre dicha vista se están realizando sobre el array original. Por ejemplo:   

# In[ ]:


C[5:8] = 12
C


# Y además hay que tener en cuenta que cualquier referencia a una vista es en realidad una referencia a los datos originales, y que las modificaciones que se realicen a través de esa referencia, se realizarán igualmente sobre el original.
# 
# Veámos esto con el siguiente ejemplo:

# In[ ]:


# C_slice referencia a las componenentes 5, 6 y 7 del array C.
C_slice = C[5:8]
C_slice


# Modificamos la componente 1 de `C_slice`:

# In[ ]:


C_slice[1] = 12345
C_slice


# Pero la componente 1 de `C_slice` es en realidad la componente 6 de `C`, así que `C` ha cambiado:

# In[ ]:


C


# Podemos incluso cambiar toda la subsecuencia, cambiando así es parte del array original:

# In[ ]:


C_slice[:] = 64
C


# Nótese la diferencia con las listas de python, en las que `l[:]` es la manera estándar de crear una *copia* de una lista `l`. En el caso de *numpy*, si se quiere realizar una copia, se ha de usar el método `copy` (por ejemplo, `C.copy()`).

# **Arrays de más dimensiones**

# El acceso a los componentes de arrays de dos o más dimensiones es similar, aunque la casuística es más variada.

# Cuando accedemos con un único índice, estamos accediendo al correspondiente subarray de esa posición. Por ejemplo, en array de dos dimensiones, con 3 filas y 3 columnas, la posición 2 es la tercera fila:

# In[ ]:


C2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
C2d[2]


# De esta manera, recursivamente, podríamos acceder a los componentes individuales de una array de cualquier dimensión. En el ejemplo anterior, el elemento de la primera fila y la tercera columna sería:

# In[ ]:


C2d[0][2]


# Normalmente no se suele usar la notación anterior para acceder a los elementos individuales, sino que se usa un único corchete con los índices separados por comas: Lo siguiente es equivalente:

# In[ ]:


C2d[0, 2]


# Veamos más ejemplos de acceso y modificación en arrays multidimensionales, en este caso con tres dimensiones.

# In[ ]:


C3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])


# Accediendo a la posición 0 obtenemos el correspondiente subarray de dos dimensiones:

# In[ ]:


C3d[0]


# Vamos a guardar una copia de de ese subarray y lo modificamos en el original con el número `42` en todas las posiciones:

# In[ ]:


old_values = C3d[0].copy()
C3d[0] = 42
C3d


# Y ahora reestablecemos los valores originales:

# In[ ]:


C3d[0] = old_values
C3d


# Como antes, podemos en este array de tres adimensiones acceder a una de sus componentes, especificando los tres índices: 

# In[ ]:


C3d[1,0,2]


# Si sólo especificamos dos de los tres índices, accedemos al correspondiente subarray unidimensional:

# In[ ]:


C3d[1, 0]


# #### Indexado usando *slices*

# In[ ]:


C2d


# Los *slicings* en arrays multidimensionales se hacen a lo largo de los correspondientes ejes. Por ejemplo, en un array bidimensional, lo haríamos sobre la secuencia de filas. 

# In[ ]:


C2d[:2]


# Pero también podríamos hacerlo en ambos ejes. Por ejemplo para obtener el subarray hasta la segunda fila y a partir de la primera columna:

# In[ ]:


C2d[:2, 1:]


# Si en alguno de los ejes se usa un índice individual, entonces se pierde una de las dimensiones:

# In[ ]:


C2d[1, :2]


# Nótese la diferencia con la operación `C2d[1:2,:2]`. Puede parecer que el resultado ha de ser el mismo, pero si se usa slicing en ambos ejes se mantiene el número de dimensiones:

# In[ ]:


C2d[1:2,:2]


# Más ejemplos:

# In[ ]:


C2d[:2, 2]


# In[ ]:


C2d[:, :1]


# Como hemos visto más arriba, podemos usar *slicing* para asignar valores a las componentes de un array. Por ejemplo

# In[ ]:


C2d[:2, 1:] = 0
C2d


# ### Indexado con booleanos

# Los arrays de booleanos se pueden usar en numpy como una forma de indexado para seleccionar determinadas componenetes en una serie de ejes. 
# 
# Veamos el siguiente ejemplo:

# In[ ]:


nombres = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])


# In[ ]:


data = np.random.randn(7, 4)
data


# Podríamos interpretar que cada fila del array `data` son datos asociados a las correspondientes personas del array `nombres`. Si ahora queremos quedarnos por ejemplos con las filas correspondientes a Bob, podemos usar indexado booleano de la siguiente manera:

# El array de booleanos que vamos a usar será:

# In[ ]:


nombres == 'Bob'


# Y el indexado con ese array, en el eje de las filas, nos dará el subarray de las filas correspondientes a Bob:

# In[ ]:


data[nombres == 'Bob']


# Podemos mezclar indexado booleano con índices concretos o con slicing en distintos ejes:

# In[ ]:


data[nombres == 'Bob', 2:]


# In[ ]:


data[nombres == 'Bob', 3]


# Para usar el indexado complementario (en el ejemplo, las filas correspondientes a las personas que no son Bob), podríamos usar el array de booleanos `nombres != 'Bob'`. Sin embargo, es más habitual usar el operador `~`:

# In[ ]:


data[~(nombres == 'Bob')]


# Incluso podemos jugar con otros operadores booleanos como `&` (and) y `|` (or), para construir indexados booleanos que combinan condiciones. 
# 
# Por ejemplo, para obtener las filas correspondiente a Bob o a Will:

# In[ ]:


mask = (nombres == 'Bob') | (nombres == 'Will')
mask


# In[ ]:


data[mask]


# Y como en los anteriores indexados, podemos usar el indexado booleano para modificar componentes de los arrays. Lo siguiente pone a 0 todos los componentes neativos de `data`:

# In[ ]:


data<0


# In[ ]:


data[data < 0] = 0
data


# Obsérvese que ahora `data<0` es un array de booleanos bidimensional con la misma estructura que el propio `data` y que por tanto tanto estamos haciendo indexado booleano sobre ambos ejes. 
# 
# Podríamos incluso fijar un valor a filas completas, usando indexado por un booleano unidimensional:

# In[ ]:


data[~(nombres == 'Joe')] = 7
data


# ### *Fancy Indexing*

# El término *fancy indexing* se usa en numpy para indexado usando arrays de enteros. Veamos un ejemplo:

# In[ ]:


arr = np.empty((8, 4))
for i in range(8):
    arr[i] = i
arr


# Indexando con `[4,3,0,6]` nos permite obtener las filas indicadas en el orden dado:

# In[ ]:


arr[[4, 3, 0, 6]]


# Si usamos más de un array para hacer *fancy indexing*, entonces toma los componentes descritos por la tupla de índices correspondiente al `zip` de los arrays de índices. Veámoslos con el siguiente ejemplo:

# In[ ]:


arr = np.arange(32).reshape((8, 4))
arr


# In[ ]:


arr[[1, 5, 7, 2], [0, 3, 1, 2]]


# Se han obtenido los elementos de índices `(1,0)`, `(5,3)`, `(7,1)` y `(2,2)`: 

# Quizás en este último caso, lo "natural" sería que nos hubiera devuelto el subarray formado por las filas 1, 5, 7 y 2 (en ese orden), y de ahí solo las columnas 0, 3 , 1 y 2 (en ese orden. 
# 
# Para obtener eso debemos hacer la siguiente operación:

# In[ ]:


arr[[1, 5, 7, 2]][:, [0, 3, 1, 2]]


# Dos observaciones importantes sobre el *fancy indexing*:
# 
# * Siempre devuelve arrays unidimensionales.
# * A diferencia del *slicing*, siempre construye una copia del array sobre el que se aplica (nunca una *vista*). 

# ### Trasposición de arrays y producto matricial

# El método `T` obtiene el array traspuesto de uno dado:

# In[ ]:


D = np.arange(15).reshape((3, 5))
D


# In[ ]:


D.T


# In[ ]:


D


# En el cálculo matricial será de mucha utilidad el método `np.dot` de numpy, que sirve tanto para calcular el producto escalar como el producto matricial. Veamos varios usos: 

# In[ ]:


E = np.random.randn(6, 3)
E


# Ejemplos de producto escalar:

# In[ ]:


np.dot(E[:,0],E[:,1]) # producto escalar de dos columnas


# In[ ]:


np.dot(E[2],E[4]) # producto escalar de dos filas


# In[ ]:


np.dot(E.T, E[:,0]) # producto de una matriz por un vector


# In[ ]:


np.dot(E.T,E)   # producto de dos matrices


# In[ ]:


np.dot(E,E.T)   # producto de dos matrices


# In[ ]:


np.dot(E.T, E[:,:1]) # producto de dos matrices


# ## Funciones universales sobre arrays (componente a componente)

# En este contexto, una función universal (o *ufunc*) es una función que actúa sobre cada componente de un array o arrays de numpy. Estas funciones son muy eficientes y se denominan *vectorizadas*. Por ejemplo:   

# In[ ]:


M = np.arange(10)
M


# In[ ]:


np.sqrt(M) # raiz cuadrada de cada componente


# In[ ]:


np.exp(M.reshape(2,5)) # exponencial de cad componente


# Existen funciones universales que actúan sobre dos arrays, ya que realizan operaciones binarias:

# In[ ]:


x = np.random.randn(8)
y = np.random.randn(8)
x,y


# In[ ]:


np.maximum(x, y)


# Existe una numerosa colección de *ufuncs* tanto unarias como bianrias. Se recomienda consultar el manual. 

# ### Expresiones condicionales vectorizadas con *where*

# Veamos cómo podemos usar un versión vectorizada de la función `if`. 
# 
# Veámoslo con un ejemplo. Supongamos que tenemos dos arrays (unidimensionales) numéricos y otro array booleano del mismo tamaño: 

# In[ ]:


xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])


# Si quisiéramos obtener el array que en cada componente tiene el valor de `xs` si el correspondiente en `cond` es `True`, o el valor de `ys` si el correspondiente en `cond` es `False`, podemos hacer lo siguiente:  

# In[ ]:


result = [(x if c else y)
          for x, y, c in zip(xarr, yarr, cond)]
result


# Sin embargo, esto tiene dos problemas: no es lo suficientemente eficiente, y además no se traslada bien a arrays multidimensionales. Afortunadamente, tenemos `np.where` para hacer esto de manera conveniente:

# In[ ]:


result = np.where(cond, xarr, yarr)
result


# No necesariamente el segundo y el tercer argumento tiene que ser arrays. Por ejemplo:

# In[ ]:


F = np.random.randn(4, 4)

F,np.where(F > 0, 2, -2)


# O una combinación de ambos. Por ejemplos, para modificar sólo las componentes positivas:

# In[ ]:


np.where(F > 0, 2, F) 


# ### Funciones estadísticas

# Algunos métodos para calcular indicadores estadísticos sobre los elementos de un array.
# 
# * `np.sum`: suma de los componentes
# * `np.mean`: media aritmética
# * `np.std` y `np.var`: desviación estándar y varianza, respectivamente.
# * `np.max` y `np.min`: máximo y mínimo, resp.
# * `np.argmin` y `np.argmax`: índices de los mínimos o máximos elementos, respectivamente.
# * `np.cumsum`: sumas acumuladas de cada componente
# 
# Estos métodos también se pueden usar como atributos de los arrays. Es decir, por ejemplo `A.sum()` o `A.mean()`.
# 
# Veamos algunos ejemplos, generando en primer lugar un array con elementos generados aleatoriamente (siguiendo una distribución normal):

# In[ ]:


G = np.random.randn(5, 4)
G


# In[ ]:


G.sum()


# In[ ]:


G.mean()


# In[ ]:


G.cumsum() # por defecto, se aplana el array y se hace la suma acumulada


# Todas estas funciones se pueden aplicar a lo largo de un eje, usando el parámetro `axis`. Por ejemplos, para calcular las medias de cada fila (es decir, recorriendo en el sentido de las columnas), aplicamos `mean` por `axis=1`:

# In[ ]:


G.mean(axis=1)


# Y la suma de cada columna (es decir, recorriendo las filas), con `sum` por `axis=0`:

# In[ ]:


G.sum(axis=0)


# Suma acumulada de cada columna:

# In[ ]:


G.cumsum(axis=0)


# Dentro de cada columna, el número de fila donde se alcanza el mínimo se puede hacer asi:

# In[ ]:


G,G.argmin(axis=0)


# ### Métodos para arrays booleanos

# In[ ]:


H = np.random.randn(50)
H


# Es bastante frecuente usar `sum` para ontar el número de veces que se cumple una condición en un array, aprovechando que `True` se identifica con 1 y `False` con 0:

# In[ ]:


(H > 0).sum() # Number of positive values


# Las funciones python `any` y `all` tienen también su correspondiente versión vectorizada. `any` se puede ver como un *or* generalizado, y `all`como un *and* generalizado:  

# In[ ]:


bools = np.array([False, False, True, False])
bools.any(),bools.all()


# Podemos comprobar si se cumple *alguna vez* una condición entre los componentes de un array, o bien si se cumple *siempre* una condición:

# In[ ]:


np.any(H>0)


# In[ ]:


np.all(H< 10)


# In[ ]:


np.any(H > 15)


# In[ ]:


np.all(H >0)


# ## Entrada y salida de arrays en ficheros

# Existen una serie de utilidades para guardar el contenido de un array en un fichero y recuperarlo más tarde. 

# Las funciones `save` y `load` hacen esto. Los arrays se almacenan en archivos con extensión *npy*.  

# In[ ]:


J = np.arange(10)
np.save('un_array', J)


# In[ ]:


np.load('un_array.npy')


# Con `savez`, podemos guardar una serie de arrays en un archivo de extensión *npz*, asociados a una serie de claves. Por ejemplo:

# In[ ]:


np.savez('array_archivo.npz', a=J, b=J**2)


# Cuando hacemos `load` sobre un archivo *npz*, cargamos un objeto de tipo diccionario, con el que podemos acceder (de manera perezosa) a los distintos arrays que se han almacenado:

# In[ ]:


arch = np.load('array_archivo.npz')
arch['b']


# In[ ]:


arch['a']


# En caso de que fuera necesario, podríamos incluso guardar incluso los datos en formato comprimido con `savez_compressed`:

# In[ ]:


np.savez_compressed('arrays_comprimidos.npz', a=arr, b=arr)


# In[ ]:


get_ipython().system('rm un_array.npy')
get_ipython().system('rm array_archivo.npz')
get_ipython().system('rm arrays_comprimidos.npz')

