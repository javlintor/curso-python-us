#!/usr/bin/env python
# coding: utf-8

# (numpy)=
# # Introducción a Numpy

# In[1]:


import numpy as np
np.random.seed(12345)
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(10, 6))
np.set_printoptions(precision=4, suppress=True)


# Numpy es quizás el paquete de computación numérica má impprtante de Python. Implementa arrays (matrices) multidimensionales de manera muy eficiente, con numerosas funcionalidades optimizadas sobre dicha estructura de datos. 
# 
# Es muy común en la comunidad python usar el alias `np` cundo importamos Numpy:

# In[2]:


import numpy as np


# Aunque en principio las listas de python podrían servir para representar array de varias dimensiones, la eficiencia de numpy es mucho mejor, al estar construido sobre una biblioteca de rutinas en lenguaje C. Además muchas de las operaciones numpy que actúan sobre todo el array, están optimizadas y permiten evitar los bucles `for`de python, que actúan más lentamente.
# 
# Lo que sigue es un ejemplo de un array de numpy unidimensional con un millón de componentes, y el análogo como lista python. 

# In[3]:


arr1 = np.arange(1000000)
list1 = list(range(1000000))


# Vamos a obtener el array resultante de multiplicar por 2 cada componente, y veamos el tiempo de CPU que se emplea. Nótese que en el caso de numpy, dicha operación se especifica simplemente como "multiplicar por 2" el array. En el caso de las listas, tenemos que usar un bucle `for` para la misma operación. Obsérvese la gran diferencia en el tiempo de ejecución:  

# In[4]:


get_ipython().run_line_magic('time', 'for _ in range(10): arr2 = arr1 * 2')
get_ipython().run_line_magic('time', 'for _ in range(10): list2 = [x * 2 for x in list1]')


# ## Arrays de Numpy

# La estructura de datos principal de numpy es el *array n-dimensional*. Como hemos dicho, numpy nos permite operar sobre los arrays en su totalidad,  especificando las operaciones como si lo hiciéramos con las componentes individuales. 

# Lo que sigue genera un array de dos dimensiones, con dos filas y tres columnas, con numeros aleatorios (obtenidos como muestras de una distribución normal de media 0 y desviación típica 1).

# In[5]:


data = np.random.randn(2, 3)
data


# Podemos por ejemplo obtener el array resultante de multiplicar cada componente del array por 10:

# In[6]:


data * 10


# O la suma de cada componente consigo mismo:

# In[7]:


data + data


# Nótese que las operaciones anteriores no cambian el array sobre el que operan:

# In[8]:


data


# Los arrays de numpy deben ser homogéneos, es decir todas sus componentes del mismo tipo. El tipo de los elementos de un array lo obtenemos con `dtype`: 

# In[9]:


data.dtype


# El método `ndim` nos da el número de dimensiones, y el método `shape` nos da el tamaño de cada dimensión, en forma de tupla. 
# 
# En este caso, tenemos que `data` es un array bidimensional, con 2 "filas" y 3 "columnas":

# In[10]:


print(data.ndim)
print(data.shape)


# En la terminología de numpy, cada una de las dimensiones se denominan "ejes" (*axis*), y se numeran consecutivamente desde 0. Por ejemplo, en un array bidimensional, el eje 0 (*axis=0*) corresponde a las filas y el eje 1 corresponde a las columnas (*axis=1*). 

# ### Creación de arrays

# La manera más fácil de crear arrays es mediante el método `np.array`. Basta con aplicarlo a cualquier dato python de tipo secuencia que sea susceptible de transformarse en un array. Por ejemplo, una lista de números se transforma en un array unidimensional: 

# In[11]:


data1 = [6, 7.5, 8, 0, 1]
arr1 = np.array(data1)
arr1


# Si le pasamos listas anidadas con una estructura correcta, podemos obtener el correspondiente array bidimensional: 

# In[12]:


data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
arr2 = np.array(data2)
arr2


# Hay otras maneras de crear arrays. Damos a continuación solo algunos ejemplos:

# In[13]:


np.zeros(10)


# In[14]:


np.ones((3, 6))


# In[15]:


np.arange(20)


# In[16]:


np.eye(7)


# Podemos usar también `reshape` para transformar entre distintas dimensiones. Por ejemplo, un uso típico es crear un array bidimensional a partir de uno unidimensional:

# In[17]:


L=np.arange(12)
M=L.reshape(3,4)
L,M


# Hasta ahora sólo hemos visto ejemplos de arrays unidimensionales y bidimensionales. Estas son las dimensiones más frecuentes, pero numpy soporta arrays con cualquier número finito de dimensiones. Por ejemplo, aquí vemos un array de tres dimensiones:

# In[18]:


np.zeros((2, 3, 2))


# ### Tipos de datos en las componentes de un array (*dtype*)

# El tipo de dato de las componentes de un array está implícito cuando se crea, pero podemos especificarlo: 

# In[19]:


arr1 = np.array([1, 2, 3], dtype=np.float64)
arr2 = np.array([1, 2, 3], dtype=np.int32)
print(arr1.dtype)
print(arr2.dtype)


# Podemos incluso convertir el `dtype` de un array que ya se ha creado, usando el método `astype`. En el ejemplo que sigue, a partir de un array de enteros, obtenemos uno de números de coma flotante:

# In[20]:


arr = np.array([1, 2, 3, 4, 5])
arr.dtype


# In[21]:


float_arr = arr.astype(np.float64)
float_arr.dtype
float_arr


# Podemos incluso pasar de coma flotante a enteros, en cuyo caso se trunca la parte decimal:

# In[22]:


arr = np.array([3.7, -1.2, -2.6, 0.5, 12.9, 10.1])
arr
arr.astype(np.int32)


# O pasar de strings a punto flotante, siempre que los strings del array tengan sentido como números:

# In[23]:


numeric_strings = np.array(['1.25', '-9.6', '42'], dtype=np.string_)
numeric_strings.astype(float)


# ### Operaciones aritméticas con arrays

# Una de las características más interesantes de numpy es la posibilidad de aplicar eficientes operaciones aritméticas "componente a componente" entre arrays, sin necesidad de usar bucles `for`. Basta con usar la correspondiente operación numérica de python. 
# 
# Veamos algunos ejemplos:

# In[24]:


A = np.array([[1., 2., 3.], [4., 5., 6.]])
B = np.array([[8.1, -22, 12.3], [6.1, 7.8, 9.2]])


# In[25]:


A * B


# In[26]:


A - B


# En principio, para poder aplicar estas operaciones entre arrays, se deben aplicar sobre arrays con las mismas dimensiones (el mismo `shape`). Es posible operar entre arrays de distintas dimensiones, con el mecanismo de *broadcasting*, pero es ésta una característica avanzada de numpy que no veremos aquí.  

# Podemos efectuar operaciones entre arrays y números (*escalares*), indicando con ello que la operación con el escalar se aplica a cada uno de los componentes del array. Vemos algunos ejemplos: 

# In[27]:


3+B


# In[28]:


1 / A


# In[29]:


A ** 0.5


# Igualmente, podemos efectuar comparaciones aritméticas entre dos arrays, obteniendo el correspondiente array de booleanos como resultado:

# In[30]:


A > B-5


# En todos los casos anteriores, nótese que estas operaciones no modifican los arrays sobre los que se aplican, sino que obtienen un nuevo array con el correspondiente resultado.

# ### Indexado y *slicing* (operaciones básicas) 

# Otra de las características más interesantes de numpy es la gran flexibilidad para acceder a las componentes de un array, o a un subconjunto del mismo. Vamos a ver a continuación algunos ejemplos básicos.

# **Arrays unidimensonales**

# Para arrays unidimensionales, el acceso es muy parecido al de listas. Por ejemplo, acceso a las componentes:

# In[31]:


C = np.arange(10)*2
C


# In[32]:


C[5]


# La operación de *slicing* en arrays es similar a la de listas. Por ejemplo:

# In[33]:


C[5:8]


# Sin embargo, hay una diferencia fundamental: en general en python, el slicing siempre crea *una copia* de la secuencia original. En numpy, el *slicing* es una *vista* de array original. Esto tiene como consecuencia que las modificaciones que se realicen sobre dicha vista se están realizando sobre el array original. Por ejemplo:   

# In[34]:


C[5:8] = 12
C


# Y además hay que tener en cuenta que cualquier referencia a una vista es en realidad una referencia a los datos originales, y que las modificaciones que se realicen a través de esa referencia, se realizarán igualmente sobre el original.
# 
# Veámos esto con el siguiente ejemplo:

# In[35]:


# C_slice referencia a las componenentes 5, 6 y 7 del array C.
C_slice = C[5:8]
C_slice


# Modificamos la componente 1 de `C_slice`:

# In[36]:


C_slice[1] = 12345
C_slice


# Pero la componente 1 de `C_slice` es en realidad la componente 6 de `C`, así que `C` ha cambiado:

# In[37]:


C


# Podemos incluso cambiar toda la subsecuencia, cambiando así es parte del array original:

# In[38]:


C_slice[:] = 64
C


# Nótese la diferencia con las listas de python, en las que `l[:]` es la manera estándar de crear una *copia* de una lista `l`. En el caso de *numpy*, si se quiere realizar una copia, se ha de usar el método `copy` (por ejemplo, `C.copy()`).

# **Arrays de más dimensiones**

# El acceso a los componentes de arrays de dos o más dimensiones es similar, aunque la casuística es más variada.

# Cuando accedemos con un único índice, estamos accediendo al correspondiente subarray de esa posición. Por ejemplo, en array de dos dimensiones, con 3 filas y 3 columnas, la posición 2 es la tercera fila:

# In[39]:


C2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
C2d[2]


# De esta manera, recursivamente, podríamos acceder a los componentes individuales de una array de cualquier dimensión. En el ejemplo anterior, el elemento de la primera fila y la tercera columna sería:

# In[40]:


C2d[0][2]


# Normalmente no se suele usar la notación anterior para acceder a los elementos individuales, sino que se usa un único corchete con los índices separados por comas: Lo siguiente es equivalente:

# In[41]:


C2d[0, 2]


# Veamos más ejemplos de acceso y modificación en arrays multidimensionales, en este caso con tres dimensiones.

# In[42]:


C3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])


# Accediendo a la posición 0 obtenemos el correspondiente subarray de dos dimensiones:

# In[43]:


C3d[0]


# Vamos a guardar una copia de de ese subarray y lo modificamos en el original con el número `42` en todas las posiciones:

# In[44]:


old_values = C3d[0].copy()
C3d[0] = 42
C3d


# Y ahora reestablecemos los valores originales:

# In[45]:


C3d[0] = old_values
C3d


# Como antes, podemos en este array de tres adimensiones acceder a una de sus componentes, especificando los tres índices: 

# In[46]:


C3d[1,0,2]


# Si sólo especificamos dos de los tres índices, accedemos al correspondiente subarray unidimensional:

# In[47]:


C3d[1, 0]


# #### Indexado usando *slices*

# In[48]:


C2d


# Los *slicings* en arrays multidimensionales se hacen a lo largo de los correspondientes ejes. Por ejemplo, en un array bidimensional, lo haríamos sobre la secuencia de filas. 

# In[49]:


C2d[:2]


# Pero también podríamos hacerlo en ambos ejes. Por ejemplo para obtener el subarray hasta la segunda fila y a partir de la primera columna:

# In[50]:


C2d[:2, 1:]


# Si en alguno de los ejes se usa un índice individual, entonces se pierde una de las dimensiones:

# In[51]:


C2d[1, :2]


# Nótese la diferencia con la operación `C2d[1:2,:2]`. Puede parecer que el resultado ha de ser el mismo, pero si se usa slicing en ambos ejes se mantiene el número de dimensiones:

# In[52]:


C2d[1:2,:2]


# Más ejemplos:

# In[53]:


C2d[:2, 2]


# In[54]:


C2d[:, :1]


# Como hemos visto más arriba, podemos usar *slicing* para asignar valores a las componentes de un array. Por ejemplo

# In[55]:


C2d[:2, 1:] = 0
C2d


# ### Indexado con booleanos

# Los arrays de booleanos se pueden usar en numpy como una forma de indexado para seleccionar determinadas componenetes en una serie de ejes. 
# 
# Veamos el siguiente ejemplo:

# In[56]:


nombres = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])


# In[57]:


data = np.random.randn(7, 4)
data


# Podríamos interpretar que cada fila del array `data` son datos asociados a las correspondientes personas del array `nombres`. Si ahora queremos quedarnos por ejemplos con las filas correspondientes a Bob, podemos usar indexado booleano de la siguiente manera:

# El array de booleanos que vamos a usar será:

# In[58]:


nombres == 'Bob'


# Y el indexado con ese array, en el eje de las filas, nos dará el subarray de las filas correspondientes a Bob:

# In[59]:


data[nombres == 'Bob']


# Podemos mezclar indexado booleano con índices concretos o con slicing en distintos ejes:

# In[60]:


data[nombres == 'Bob', 2:]


# In[61]:


data[nombres == 'Bob', 3]


# Para usar el indexado complementario (en el ejemplo, las filas correspondientes a las personas que no son Bob), podríamos usar el array de booleanos `nombres != 'Bob'`. Sin embargo, es más habitual usar el operador `~`:

# In[62]:


data[~(nombres == 'Bob')]


# Incluso podemos jugar con otros operadores booleanos como `&` (and) y `|` (or), para construir indexados booleanos que combinan condiciones. 
# 
# Por ejemplo, para obtener las filas correspondiente a Bob o a Will:

# In[63]:


mask = (nombres == 'Bob') | (nombres == 'Will')
mask


# In[64]:


data[mask]


# Y como en los anteriores indexados, podemos usar el indexado booleano para modificar componentes de los arrays. Lo siguiente pone a 0 todos los componentes neativos de `data`:

# In[65]:


data<0


# In[66]:


data[data < 0] = 0
data


# Obsérvese que ahora `data<0` es un array de booleanos bidimensional con la misma estructura que el propio `data` y que por tanto tanto estamos haciendo indexado booleano sobre ambos ejes. 
# 
# Podríamos incluso fijar un valor a filas completas, usando indexado por un booleano unidimensional:

# In[67]:


data[~(nombres == 'Joe')] = 7
data


# ### *Fancy Indexing*

# El término *fancy indexing* se usa en numpy para indexado usando arrays de enteros. Veamos un ejemplo:

# In[68]:


arr = np.empty((8, 4))
for i in range(8):
    arr[i] = i
arr


# Indexando con `[4,3,0,6]` nos permite obtener las filas indicadas en el orden dado:

# In[69]:


arr[[4, 3, 0, 6]]


# Si usamos más de un array para hacer *fancy indexing*, entonces toma los componentes descritos por la tupla de índices correspondiente al `zip` de los arrays de índices. Veámoslos con el siguiente ejemplo:

# In[70]:


arr = np.arange(32).reshape((8, 4))
arr


# In[71]:


arr[[1, 5, 7, 2], [0, 3, 1, 2]]


# Se han obtenido los elementos de índices `(1,0)`, `(5,3)`, `(7,1)` y `(2,2)`: 

# Quizás en este último caso, lo "natural" sería que nos hubiera devuelto el subarray formado por las filas 1, 5, 7 y 2 (en ese orden), y de ahí solo las columnas 0, 3 , 1 y 2 (en ese orden. 
# 
# Para obtener eso debemos hacer la siguiente operación:

# In[72]:


arr[[1, 5, 7, 2]][:, [0, 3, 1, 2]]


# Dos observaciones importantes sobre el *fancy indexing*:
# 
# * Siempre devuelve arrays unidimensionales.
# * A diferencia del *slicing*, siempre construye una copia del array sobre el que se aplica (nunca una *vista*). 

# ### Trasposición de arrays y producto matricial

# El método `T` obtiene el array traspuesto de uno dado:

# In[73]:


D = np.arange(15).reshape((3, 5))
D


# In[74]:


D.T


# In[75]:


D


# En el cálculo matricial será de mucha utilidad el método `np.dot` de numpy, que sirve tanto para calcular el producto escalar como el producto matricial. Veamos varios usos: 

# In[76]:


E = np.random.randn(6, 3)
E


# Ejemplos de producto escalar:

# In[77]:


np.dot(E[:,0],E[:,1]) # producto escalar de dos columnas


# In[78]:


np.dot(E[2],E[4]) # producto escalar de dos filas


# In[79]:


np.dot(E.T, E[:,0]) # producto de una matriz por un vector


# In[80]:


np.dot(E.T,E)   # producto de dos matrices


# In[81]:


np.dot(E,E.T)   # producto de dos matrices


# In[82]:


np.dot(E.T, E[:,:1]) # producto de dos matrices


# ## Funciones universales sobre arrays (componente a componente)

# En este contexto, una función universal (o *ufunc*) es una función que actúa sobre cada componente de un array o arrays de numpy. Estas funciones son muy eficientes y se denominan *vectorizadas*. Por ejemplo:   

# In[83]:


M = np.arange(10)
M


# In[84]:


np.sqrt(M) # raiz cuadrada de cada componente


# In[85]:


np.exp(M.reshape(2,5)) # exponencial de cad componente


# Existen funciones universales que actúan sobre dos arrays, ya que realizan operaciones binarias:

# In[86]:


x = np.random.randn(8)
y = np.random.randn(8)
x,y


# In[87]:


np.maximum(x, y)


# Existe una numerosa colección de *ufuncs* tanto unarias como bianrias. Se recomienda consultar el manual. 

# ### Expresiones condicionales vectorizadas con *where*

# Veamos cómo podemos usar un versión vectorizada de la función `if`. 
# 
# Veámoslo con un ejemplo. Supongamos que tenemos dos arrays (unidimensionales) numéricos y otro array booleano del mismo tamaño: 

# In[88]:


xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])


# Si quisiéramos obtener el array que en cada componente tiene el valor de `xs` si el correspondiente en `cond` es `True`, o el valor de `ys` si el correspondiente en `cond` es `False`, podemos hacer lo siguiente:  

# In[89]:


result = [(x if c else y)
          for x, y, c in zip(xarr, yarr, cond)]
result


# Sin embargo, esto tiene dos problemas: no es lo suficientemente eficiente, y además no se traslada bien a arrays multidimensionales. Afortunadamente, tenemos `np.where` para hacer esto de manera conveniente:

# In[90]:


result = np.where(cond, xarr, yarr)
result


# No necesariamente el segundo y el tercer argumento tiene que ser arrays. Por ejemplo:

# In[91]:


F = np.random.randn(4, 4)

F,np.where(F > 0, 2, -2)


# O una combinación de ambos. Por ejemplos, para modificar sólo las componentes positivas:

# In[92]:


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

# In[93]:


G = np.random.randn(5, 4)
G


# In[94]:


G.sum()


# In[95]:


G.mean()


# In[96]:


G.cumsum() # por defecto, se aplana el array y se hace la suma acumulada


# Todas estas funciones se pueden aplicar a lo largo de un eje, usando el parámetro `axis`. Por ejemplos, para calcular las medias de cada fila (es decir, recorriendo en el sentido de las columnas), aplicamos `mean` por `axis=1`:

# In[97]:


G.mean(axis=1)


# Y la suma de cada columna (es decir, recorriendo las filas), con `sum` por `axis=0`:

# In[98]:


G.sum(axis=0)


# Suma acumulada de cada columna:

# In[99]:


G.cumsum(axis=0)


# Dentro de cada columna, el número de fila donde se alcanza el mínimo se puede hacer asi:

# In[100]:


G,G.argmin(axis=0)


# ### Métodos para arrays booleanos

# In[101]:


H = np.random.randn(50)
H


# Es bastante frecuente usar `sum` para ontar el número de veces que se cumple una condición en un array, aprovechando que `True` se identifica con 1 y `False` con 0:

# In[102]:


(H > 0).sum() # Number of positive values


# Las funciones python `any` y `all` tienen también su correspondiente versión vectorizada. `any` se puede ver como un *or* generalizado, y `all`como un *and* generalizado:  

# In[103]:


bools = np.array([False, False, True, False])
bools.any(),bools.all()


# Podemos comprobar si se cumple *alguna vez* una condición entre los componentes de un array, o bien si se cumple *siempre* una condición:

# In[104]:


np.any(H>0)


# In[105]:


np.all(H< 10)


# In[106]:


np.any(H > 15)


# In[107]:


np.all(H >0)


# ## Entrada y salida de arrays en ficheros

# Existen una serie de utilidades para guardar el contenido de un array en un fichero y recuperarlo más tarde. 

# Las funciones `save` y `load` hacen esto. Los arrays se almacenan en archivos con extensión *npy*.  

# In[108]:


J = np.arange(10)
np.save('un_array', J)


# In[109]:


np.load('un_array.npy')


# Con `savez`, podemos guardar una serie de arrays en un archivo de extensión *npz*, asociados a una serie de claves. Por ejemplo:

# In[110]:


np.savez('array_archivo.npz', a=J, b=J**2)


# Cuando hacemos `load` sobre un archivo *npz*, cargamos un objeto de tipo diccionario, con el que podemos acceder (de manera perezosa) a los distintos arrays que se han almacenado:

# In[111]:


arch = np.load('array_archivo.npz')
arch['b']


# In[112]:


arch['a']


# En caso de que fuera necesario, podríamos incluso guardar incluso los datos en formato comprimido con `savez_compressed`:

# In[113]:


np.savez_compressed('arrays_comprimidos.npz', a=arr, b=arr)


# In[114]:


get_ipython().system('rm un_array.npy')
get_ipython().system('rm array_archivo.npz')
get_ipython().system('rm arrays_comprimidos.npz')

