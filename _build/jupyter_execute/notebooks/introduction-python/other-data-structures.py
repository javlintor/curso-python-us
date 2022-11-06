#!/usr/bin/env python
# coding: utf-8

# # Otras estructuras de datos
# 
# Hasta ahora hemos introducido las listas de Python como estructura que nos permite manipular conjutos de datos. En esta sección veremos otras estructuras de datos e investigaremos cúando es conveniente usar cada una, en concreto vamos a ver
# - tuplas 
# - diccionarios 
# - conjuntos

# ---
# ## Tuplas
# 
# Las **tuplas** son similares a las listas en el sentido de que nos permiten guardar un número arbitrario de objetos y acceder a los mismos mediante índices, es decir, son objetos secuenciales. Para definir una tupla utilizamos paréntesis `()`

# In[21]:


foo = (1, "b")


# Al igual que con las listas, podemos incluir expresiones que se evaluarán antes de formar la tupla

# In[ ]:


bar = (1 is None, "fjkdsljfd".islower(), 2 in [1, 4, 5, 3])


# In[ ]:


bar


# In[ ]:


isinstance(bar, tuple)


# In[ ]:


type(foo)


# Las tuplas son utilizadas para guardar una colección de datos en una estructura simple e **inmutable**, es decir, no podremos modificarlas una vez sean creadas: ni reemplezar, añadir o borrar sus elementos. Dicho de otro modo, es un único objeto formado por distintas partes más que una colección de distintos objetos como una lista. 

# :::{exercise}
# :label: other-data-structures-tuples-inmmutable
# 
# ¿Qué error obtenemos al intentar modificar un objeto de una tupla?
# 
# :::

# :::{exercise}
# :label: other-data-structures-tuples-inmmutable-2
# 
# Las tuplas son objetos inmutables pero, ¿pueden los objetos que forman la tupla ser mutables? 
# 
# :::

# Por consistencia, existen las tuplas de longitud 0 y 1

# In[ ]:


zero_tuple = ()
one_tuple = (5,) # notemos la coma


# Aunque podemos utilizar índices numéricos para acceder a los elementos de la tupla, es más común *deshacer* la tupla en variables 

# In[ ]:


holding = ('GOOG', 100, 490.10)
address = ('www.python.org', 80)

name, shares, price = holding
host, port = address


# > Esta misma sintaxis se puede utilizar para hacer varias asignaciones a la vez
# ```
# a, b = 1, None
# ```

# In[ ]:


isinstance(1, (bool, int))


# ---
# ## Trabajando con secuencias 
# 
# Ya hemos visto tres tipos secuenciales: cadenas, listas y tuplas. Vamos a dedicar un apartado a repasar las principales operaciones que podemos realizar con objetos secuenciales

# ### Comprobar pertenencia 
# 
# Lo haremos a través del operador `in` y su negación `not in`

# In[ ]:


# con tuplas
x = (1, 3, 5)
3 in x


# In[ ]:


# con cadenas 
"cat" in "the cat in the hat"
True


# In[ ]:


# con listas 
[1, 2] in [1, 2, 3, 4]


# In[ ]:


[1, 2] in [None, [1, 2], None]


# ### Obtener el índice de la primera instancia de un objeto
# 
# Mediante el método `index`

# In[ ]:


"cat cat cat".index("cat")


# In[ ]:


[1, 2, "moo"].index("m")


# ### Contar el número de ocurrencias 
# 
# Utilizaremos el método `count`

# In[ ]:


"the cat in the hat".count("h")


# ### Indexado y *slicing*
# 
# Como ya hemos visto, podemos acceder a objetos individuales utilizando in índice entero que empieza en cero. Este índice puede ser negativo si queremos buscar desde el final de la secuencia 

# In[ ]:


l = [1, 1, 2, 3, 5, 8, 13, 21]
l[-1]


# Podemos ir más allá y pedir un subconjunto de la sucencia con las operaciones de *slicing*, cuya sintaxis básica es `seq[start:stop:step]`

# In[ ]:


seq = "abcdefg"
seq[0:4:1]


# In[ ]:


seq[::2]


# Por defecto, `start=0`, `stop=len(seq)` y `step=1`. Si utilizamos valores negativos para `step` invertiremos el orden de la secuencia

# In[ ]:


seq[::-1]


# Aunque la sintaxis con `:` es la más frecuente, está bien saber que en Python existe el objeto de tipo `slice` para definir nuestra selección de forma independiente a la secuencia. Para ello utilizamos el tipo `slice` con los tres argumentos que hemos visto: `start`, `stop` y `step`.

# In[ ]:


reverse = slice(None, None, -1)


# In[ ]:


type(reverse)


# :::{exercise}
# :label: other-data-structures-slices
# 
# ¿Qué error obtenemos cuando intentamos acceder a un índice que no existe para una secuencia?
# 
# :::

# :::{exercise}
# :label: other-data-structures-slices-2
# 
# Considera la siguiente tupla
# 
# ```
# x = (0, 2, 4, 6, 8)
# ```
# 
# Indexa o utiliza slides para obtener
# 1. `0`
# 2. `8`
# 3. `(2, 4, 6)`
# 4. `(4,)`
# 5. `4`
# 6. `4` utilizando un índice negativo
# 7. `(6, 8)`
# 8. `(2, 6)`
# 9. `(8, 6, 4, 2)`
# 
# :::

# :::{exercise}
# :label: other-data-structures-slices-3
# 
# Dada una tupla `x` que contenga el número `5`, escribe las instrucciones necesarias para reemplazar la primera instancia de `5` por `-5`.
# 
# :::

# :::{exercise}
# :label: other-data-structures-slices-4
# 
# Dada una secuencia `seq` y un índice negativo `neg_index`, escribe la fórmula que nos daría el índice positivo correspondiente. 
# 
# :::

# ---
# ## Diccionarios
# 
# 

# Un **diccionario** es un objeto que nos permite guardar campos informados mediante una clave. Para crearlos escribimos pares de clave - valor separados por `:` entre corchetes 

# In[22]:


prices = {
    "GOOG": 490.1, 
    "AAPL": 123.5, 
    "IBM": 91.5, 
    "MSFT": 52.13
}

# diccionario vacío 
empty_dict = {}    # también se crea con dict()


# Se pueden crear también diccionarios a partir del constructor `dict`, que acepta un iterable de pares de clave-valor empaquetados en una secuencia.   

# In[18]:


fruit_or_veggie = dict([("apple", "fruit"), ("carrot", "vegetable")])


# Para acceder a un valor del diccionario a través de la clave se pueden utilizar corchetes `[]` con la clave entre comillas

# In[23]:


prices["GOOG"]


# aunque es más recomendable utilizar el método `get`, ya que si la clave buscada no se encuentra devuelve un `None` o el valor por defecto que le indiquemos, en lugar de levantar un error tipo `KeyError`. 

# In[28]:


prices.get("GOOG")


# In[29]:


prices.get("AMZ", 0.0)    # devuelve 0.0 si no encuentra la clave AMZ


# In[31]:


prices["AMZ"]


# Los diccionarios son objetos **mutables**, podemos añadir elementos directamente

# In[50]:


prices["AMZ"] = 90.98
prices


# para ello también tenemos el método `update`, que acepta otro diccionario o un iterable de pares clave-valor.

# In[42]:


fruit_or_veggie.update([("grape", "fruit"), ("onion", "vegetable")])
fruit_or_veggie


# para borrar un elemento utilizamos `del` (leventa error si no encuentra la clave) o el método `pop` como vimos en listas.

# In[51]:


del prices["AMZ"]


# In[44]:


prices


# :::{exercise}
# :label: other-data-structures-dict
# 
# Dada la siguiente tupla de nombres 
# 
# ```
# ("Alicia", "Eva", "Manolo", "Virginia")
# ```
# 
# y sus correspondientes calificaciones 
# 
# ```
# (5.2, 9.1, 7.2, 4.9)
# ```
# 
# crea un diccionario que mapee nombres con calificaciones. Luego, actualiza la nota de Virginia a un `5.0`. Finalmente, añade a Alberto, que ha cateado con un `2.7`.
# 
# :::

# > Un diccionario **puede almacenar cualquier tipo de objeto**, pero las claves deben ser siempre **inmutables**, o más generalmente, **hasheables**. 

# In[57]:


example_dict = {
    -1: 10, 
    "moo": True,
    (1, 2): print,
    3.4: "cow", # es altamente no recomendable usar floats como claves 
    False:[] 
}


# :::{exercise}
# :label: other-data-structures-dict-2
# 
# ¿Qué tipo de error obtenemos al crear un diccionario con una clave que sea mutable?
# 
# :::

# Cuando iteramos sobre un diccionario, se utilizan las claves como referencia. Por ejemplo 

# In[58]:


example_dict = {"key1": "value1", "key2": "value2", "key3": "value3"}


# In[59]:


list(example_dict)


# In[60]:


"value1" in example_dict


# In[62]:


len(example_dict) # nos da el número de claves


# Para acceder a los valores de un diccionario, hay que invocar al método `values`. Si queremos los pares clave-valor como duplas, llamamos a `items`

# In[64]:


list(example_dict.values())


# In[65]:


list(example_dict.items())


# ---
# ## Comparación complejidad computacional 
# 
# Vamos a comparar las diferentes estructuras de datos que hemos visto en cuanto a su tiempo de cómputo para diferentes tareas

# ### Pertenencia
# 

# In[11]:


import time
import numpy as np

def get_membership_time_from_range(i):
    iterable = range(i)
    execution_time = get_membership_time_from_iterable(i - 1, iterable)
    return execution_time

def get_membership_time_from_list(i):
    iterable = list(range(i))
    execution_time = get_membership_time_from_iterable(i - 1, iterable)
    return execution_time

def get_membership_time_from_set(i):
    iterable = set(range(i))
    execution_time = get_membership_time_from_iterable(i - 1, iterable)
    return execution_time

def get_membership_time_from_tuple(i):
    iterable = tuple(range(i))
    execution_time = get_membership_time_from_iterable(i - 1, iterable)
    return execution_time

def get_membership_time_from_iterable(i, iterable, repeat=10):
    execution_times = []
    for _ in range(repeat):
        start = time.time()
        i in iterable
        end = time.time()
        execution_time = end - start
        execution_times.append(execution_time)
    mean_execution_time = np.mean(execution_times)
    return mean_execution_time


# In[19]:


import matplotlib.pyplot as plt

n = [10**i for i in range(8)]
t_range = [get_membership_time_from_range(i) for i in n]
t_list = [get_membership_time_from_list(i) for i in n]
t_set = [get_membership_time_from_set(i) for i in n]
t_tuple = [get_membership_time_from_tuple(i) for i in n]


# In[20]:


get_ipython().run_line_magic('config', 'InlineBackend.figure_format = "retina"')

fig, ax = plt.subplots()
ax.plot(n, t_list, "o-", label="list")
ax.plot(n, t_range, "o-", label="range")
ax.plot(n, t_set, "o-", label="set")
ax.plot(n, t_tuple, "o-", label="tuple")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("tamaño iterable")
ax.set_ylabel("tiempo (s)")
ax.set_title("Tiempo de cómputo en verificar pertenencia")
ax.grid(True)
ax.legend()
fig.show()


# In[ ]:




