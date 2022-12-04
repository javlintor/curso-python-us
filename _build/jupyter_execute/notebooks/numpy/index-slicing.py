#!/usr/bin/env python
# coding: utf-8

# # Indexado, Slicing y operaciones básicas

# Vamos a explorar más a fondo la diferentes formas que tenemos de acceder y operar con componentes de un array multidimensional.

# In[1]:


import numpy as np


# ---
# ## Indexado y *slicing* 

# Otra de las características más interesantes de numpy es la gran flexibilidad para acceder a las componentes de un array, o a un subconjunto del mismo. Vamos a ver a continuación algunos ejemplos básicos.

# **Arrays unidimensonales**

# Para arrays unidimensionales, el acceso es muy parecido al de listas. Por ejemplo, acceso a las componentes:

# In[ ]:


v = np.arange(10)


# In[ ]:


v[5]


# La operación de *slicing* en arrays es similar a la de listas. Por ejemplo:

# In[ ]:


v[5:8]


# Sin embargo, hay una diferencia fundamental: en general en python, el slicing siempre crea *una copia* de la secuencia original (aunque no de los elementos) a la hora de hacer asignaciones. En numpy, el *slicing* es una *vista* de array original. Esto tiene como consecuencia que **las modificaciones que se realicen sobre dicha vista se están realizando sobre el array original**. Por ejemplo:   

# In[ ]:


l = list(range(10))
l_slice = l[5:8]
v_slice = v[5:8]
l_slice[:] = [12, 12, 12]
v_slice[:] = 12


# In[ ]:


print(l)
print(v)


# Y además hay que tener en cuenta que cualquier referencia a una vista es en realidad una referencia a los datos originales, y que las modificaciones que se realicen a través de esa referencia, se realizarán igualmente sobre el original.
# 
# Veámos esto con el siguiente ejemplo:

# Modificamos la componente 1 de `v_slice`:

# In[ ]:


v_slice[1] = 12345
print(v_slice)


# Pero la componente 1 de `C_slice` es en realidad la componente 6 de `C`, así que `C` ha cambiado:

# In[ ]:


print(v)


# Nótese la diferencia con las listas de python, en las que `l[:]` es la manera estándar de crear una *copia* de una lista `l`. En el caso de *numpy*, si se quiere realizar una copia, se ha de usar el método `copy` (por ejemplo, `C.copy()`).

# **Arrays de más dimensiones**

# El acceso a los componentes de arrays de dos o más dimensiones es similar, aunque la casuística es más variada.

# Cuando accedemos con un único índice, estamos accediendo al correspondiente subarray de esa posición. Por ejemplo, en array de dos dimensiones, con 3 filas y 3 columnas, la posición 2 es la tercera fila:

# In[5]:


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
C3d


# Accediendo a la posición 0 obtenemos el correspondiente subarray de dos dimensiones:

# In[ ]:


C3d[0]


# Similar a la función `enumerate` de Python, tenemos la función `np.ndenumearte` para iterar con los elementos del array y su índice

# In[ ]:


[i for i in np.ndenumerate(C3d)]


# Vamos a guardar una copia de de ese subarray y lo modificamos en el original con el número `42` en todas las posiciones:

# In[ ]:


old_values = C3d[0].copy()
C3d[0] = 42
C3d


# Y ahora reestablecemos los valores originales:

# In[ ]:


C3d[0] = old_values
C3d


# :::{exercise}
# :label: introduction-numpy-indexing
# 
# Devuelve el número 813 indexando el array `np.arange(2100).reshape((25, 6, 7, 2))`. 
# 
# :::

# ### Indexado usando *slices*

# In[6]:


C2d


# Los *slicings* en arrays multidimensionales se hacen a lo largo de los correspondientes ejes. Por ejemplo, en un array bidimensional, lo haríamos sobre la secuencia de filas. 

# In[7]:


C2d[:2]


# Pero también podríamos hacerlo en ambos ejes. Por ejemplo para obtener el subarray hasta la segunda fila y a partir de la primera columna:

# In[8]:


C2d[:2, 1:]


# Si en alguno de los ejes se usa un índice individual, entonces se pierde una de las dimensiones:

# In[ ]:


C2d[1, :2]


# Nótese la diferencia con la operación `C2d[1:2,:2]`. Puede parecer que el resultado ha de ser el mismo, pero si se usa slicing en ambos ejes se mantiene el número de dimensiones:

# In[ ]:


C2d[1:2,:2]


# Más ejemplos:

# In[10]:


C2d


# In[9]:


C2d[:2, 2]


# In[ ]:


C2d[:, :, :, :, :1]


# Como hemos visto más arriba, podemos usar *slicing* para asignar valores a las componentes de un array. Por ejemplo

# In[ ]:


C2d[:2, 1:] = 0
C2d


# Finalmente, notemos que podemos usar cualquier `slice` de Python para arrays

# In[ ]:


slice_1 = slice(2, 0, -1)
slice_2 = slice(0, 3, 2)


# In[ ]:


C2d[slice_1, slice_2]


# :::{exercise}
# :label: index-slicing-3x4x2
# 
# Crea un array tridimensional de dimensiones $(3, 4, 2)$ y obtén el subarray indicada en la figura (`shape = (1, 2)`). Obtén también un subarray a tu elección de dimensiones $(2, 3, 1)$.
# 
# <div style="display: flex; align-items: center;
# justify-content: center;">
#     <img style="width: 100px; height: 100px;" src="https://drive.google.com/uc?id=1HEtbq_Y1YVh6jscdHEhYYz-iM5FNMyJP"/>
# </div>
# 
# :::

# In[11]:


arr = np.arange(3*4*2).reshape((3, 4, 2))


# ### Indexado con booleanos

# Los arrays de booleanos se pueden usar en numpy como una forma de indexado para seleccionar determinadas componenetes en una serie de ejes. 
# 
# Veamos el siguiente ejemplo:

# In[43]:


nombres = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])


# In[44]:


nombres.shape


# In[45]:


rng = np.random.default_rng()
data = rng.normal(0, 1, (7, 4))
data


# In[46]:


data[nombres == "Bob"]


# Podríamos interpretar que cada fila del array `data` son datos asociados a las correspondientes personas del array `nombres`. Si ahora queremos quedarnos por ejemplos con las filas correspondientes a Bob, podemos usar indexado booleano de la siguiente manera:

# El array de booleanos que vamos a usar será:

# In[34]:


nombres == 'Bob'


# Y el indexado con ese array, en el eje de las filas, nos dará el subarray de las filas correspondientes a Bob:

# In[35]:


data[nombres == 'Bob']


# Podemos mezclar indexado booleano con índices concretos o con slicing en distintos ejes:

# In[36]:


data[nombres == 'Bob', 2:]


# In[37]:


data[nombres == 'Bob', 3]


# Para usar el indexado complementario (en el ejemplo, las filas correspondientes a las personas que no son Bob), podríamos usar el array de booleanos `nombres != 'Bob'`. Sin embargo, es más habitual usar el operador `~`:

# In[38]:


data[~(nombres == 'Bob')]


# Incluso podemos jugar con otros operadores booleanos como `&` (and) y `|` (or), para construir indexados booleanos que combinan condiciones. 
# 
# Por ejemplo, para obtener las filas correspondiente a Bob o a Will:

# In[39]:


mask = (nombres == 'Bob') | (nombres == 'Will')
mask


# In[40]:


data[mask]


# Y como en los anteriores indexados, podemos usar el indexado booleano para modificar componentes de los arrays. Lo siguiente pone a 0 todos los componentes neativos de `data`:

# In[41]:


data < 0


# In[42]:


data[data < 0]


# In[ ]:


data[data < 0] = 0
data


# Obsérvese que ahora `data < 0` es un array de booleanos bidimensional con la misma estructura que el propio `data` y que por tanto tanto estamos haciendo indexado booleano sobre ambos ejes. 
# 
# Podríamos incluso fijar un valor a filas completas, usando indexado por un booleano unidimensional:

# In[ ]:


data[~(nombres == 'Joe')] = 7
data


# In[ ]:





# :::{exercise}
# :label: index-slicing-bool
# 
# Devuelve las filas de `data` correspondientes a aquellos nombres que empiecen por "B" o "J". Puedes utilizar la función `np.char.startswith`.
# 
# :::

# In[ ]:


# Noooo
[nombre for nombre in nombres if nombre.strartswith("")]


# In[48]:


mask = np.char.startswith(nombres, "B")\
    | np.char.startswith(nombres, "J")

data[mask]


# :::{exercise}
# :label: index-slicing-flip
# 
# Crea una función `flip` que tome como inputs un array `arr` y un número entero positivo `i` e *invierta* el eje i-ésimo, es decir, si la dimensión del eje $i$ vale $d_i$, la transformación lleva el elemento con índice $(x_1, \dots, x_i, \dots, x_n)$ en $(x_1, \dots, x_i^*, \dots, x_n)$ donde $x_i + x_i^* = d_i + 1$ 
# 
# Por ejemplo, 
# 
# ```
# arr = np.arange(9).reshape((3, 3))
# arr
# >>>
# [[0 1 2]
#  [3 4 5]
#  [6 7 8]]
#  
# flip(arr, 1)
# >>> 
# [[2 1 0]
#  [5 4 3]
#  [8 7 6]]
# ```
# 
# :::

# :::{solution} index-slicing-flip
# :class: dropdown
# 
# ```
# def flip(arr: np.ndarray, i: int):
#     default_slice = slice(None)
#     reverse_slice = slice(None, None, -1)
#     slices_gen = (reverse_slice if j == i else default_slice for j in range(arr.ndim))
#     slices = tuple(slices_gen)
#     return arr[slices]
# ```
# 
# :::
