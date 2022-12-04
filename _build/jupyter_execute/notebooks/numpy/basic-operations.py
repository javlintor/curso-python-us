#!/usr/bin/env python
# coding: utf-8

# # Operaciones Básicas

# ---
# ## Trasposición de arrays y producto matricial

# El método `T` obtiene el array traspuesto de uno dado:

# In[ ]:


D = np.arange(15).reshape((3, 5))
print(D)


# In[ ]:


print(D.T)


# En el cálculo matricial será de mucha utilidad el método `np.dot` de numpy, que sirve tanto para calcular el producto escalar como el producto matricial. Veamos varios usos: 

# In[ ]:


E = rng.normal(0, 1, (6, 3))
E


# Ejemplos de producto escalar:

# In[ ]:


np.dot(E[:, 0], E[:, 1]) # producto escalar de dos columnas


# In[ ]:


np.dot(E[2],E[4]) # producto escalar de dos filas


# In[ ]:


np.dot(E, E[0]) # producto de una matriz por un vector


# In[ ]:


np.dot(E.T, E)   # producto de dos matrices


# Existe otro operador `matmul` (o su versión con el operador `@`) que también multiplica matrices. Se diferencian cuando los arrays con de más de dos dimensiones ya  

# In[ ]:


A = np.arange(3*7*4*5).reshape(3, 7, 4, 5)
B = np.arange(3*7*5*6).reshape(3, 7, 5, 6)


# In[ ]:


np.dot(A, B).shape


# `np.dot(A, B)[x1, x2, x3, y1, y2, y3] = A[x1, x2, x3, :].dot(B[y1, y2, :, y3])`

# In[ ]:


np.matmul(A, B).shape # similar a A @ B 


# La diferencia radica en que `dot` el producto escalara del último eje de A con el penúltimo de B para cada combinación de dimensiones y `matmul` considera los arrays como *arrays de matrices*, donde las dos últimas dimensiones son la parte matricial. 

# ---
# ## Funciones universales sobre arrays (componente a componente)
# 
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


x = rng.normal(0, 1, 8)
y = rng.normal(0, 1, 8)
x, y


# In[ ]:


np.maximum(x, y)


# In[ ]:


x.max()


# ---
# ## Expresiones condicionales vectorizadas con *where*

# Veamos cómo podemos usar un versión vectorizada de la función `if`. 
# 
# Veámoslo con un ejemplo. Supongamos que tenemos dos arrays (unidimensionales) numéricos y otro array booleano del mismo tamaño: 

# In[ ]:


xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
mask = np.array([True, False, True, True, False])


# Si quisiéramos obtener el array que en cada componente tiene el valor de `xarr` si el correspondiente en `mask` es `True`, o el valor de `yarr` si el correspondiente en `cond` es `False`, podemos hacer lo siguiente:  

# In[ ]:


result = [(x if c else y) for x, y, c in zip(xarr, yarr, cond)]
result


# Sin embargo, esto tiene dos problemas: no es lo suficientemente eficiente, y además no se traslada bien a arrays multidimensionales. Afortunadamente, tenemos `np.where` para hacer esto de manera conveniente:

# In[ ]:


result = np.where(mask, xarr, yarr)
result


# No necesariamente el segundo y el tercer argumento tiene que ser arrays. Por ejemplo:

# In[ ]:


F = rng.normal(0, 1, (4, 4))

F, np.where(F > 0, 2, -2)


# O una combinación de ambos. Por ejemplos, para modificar sólo las componentes positivas:

# In[ ]:


np.where(F > 0, 2, F) 


# También existe la función `np.select` para concatenar varias máscaras consecutivas. 

# In[ ]:


np.select(
    [np.abs(F) > 2, np.abs(F) > 1],
    ["Poco probable", "Algo probable"], 
    "Frecuente"
)


# :::{exercise}
# :label: basic-operations-masks
# 
# Crea una función que transforme un array para aplicar elemento a elemento la siguiente función 
# 
# $$
#  f(x) = \begin{cases}
#         exp(x/2)  & \text{si } x < 0 \\
#         1-x & \text{si } 0 \leq x \leq 1 \\
#         0 & \text{si } x > 1
#         \end{cases}
# $$
# 
# :::

# :::{solution} basic-operations-masks
# :class: dropdown
# 
# ```
# def fun(arr: np.ndarray):
#     ret = np.select(
#         [arr < 0, arr <= 1], 
#         [np.exp(arr / 2), 1 - arr], 
#         0
#     )
#     return ret
# ```
# 
# :::

# ---
# ## Funciones estadísticas

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


G = rng.normal(0, 1, (5, 4))
G


# In[ ]:


G.sum()


# In[ ]:


G.mean()


# In[ ]:


G.cumsum() # por defecto, se aplana el array y se hace la suma acumulada


# Todas estas funciones se pueden aplicar a lo largo de un eje, usando el parámetro `axis`. Por ejemplos, para calcular las medias de cada fila (es decir, recorriendo en el sentido de las columnas), aplicamos `mean` por `axis=1`:

# In[ ]:


print(G)


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


G, G.argmin(axis=0)


# ---
# ## Métodos para arrays booleanos

# In[ ]:


H = rng.normal(0, 1, 50)
H


# Es bastante frecuente usar `sum` para ontar el número de veces que se cumple una condición en un array, aprovechando que `True` se identifica con 1 y `False` con 0:

# In[ ]:


(H > 0).sum() # Number of positive values


# Las funciones python `any` y `all` tienen también su correspondiente versión vectorizada. `any` se puede ver como un *or* generalizado, y `all`como un *and* generalizado:  

# In[ ]:


bools = np.array([False, False, True, False])
bools.any(), bools.all()


# Podemos comprobar si se cumple *alguna vez* una condición entre los componentes de un array, o bien si se cumple *siempre* una condición:

# In[ ]:


np.any(H > 0)


# In[ ]:


np.all(H < 10)


# In[ ]:


np.any(H > 15)


# In[ ]:


np.all(H > 0)


# ---
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


# In[ ]:


list(arch)


# En caso de que fuera necesario, podríamos incluso guardar incluso los datos en formato comprimido con `savez_compressed`:

# In[ ]:


np.savez_compressed('arrays_comprimidos.npz', a=J, b=J**2)


# In[ ]:


get_ipython().system('ls -lah')


# In[ ]:


get_ipython().system('rm un_array.npy')
get_ipython().system('rm array_archivo.npz')
get_ipython().system('rm arrays_comprimidos.npz')

