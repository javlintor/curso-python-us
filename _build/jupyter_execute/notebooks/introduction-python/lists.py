#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/javlintor/curso-python-us/blob/main/notebooks/introduction-python/lists.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Listas
# 
# Las **listas** son un tipo de objeto que nos permite guardar una secuencia de otros objetos, actualizarlos, añadir nuevos y borrarlos. A diferencia de todos los tipos que hemos visto hasta ahora, las listas con objetos **mutables**, es decir, pueden cambiar durante la ejecución de nuestro programa. 
# 
# Para crear una lista en Python usamos corchetes y escribimos sus elementos separados por comas `[item1, item2, ..., itemN]`. Los objetos de una lista **no tienen por qué ser del mismo tipo de objeto**

# In[1]:


[3.5, None, "foo"]


# In[2]:


type([1, 2, 3])


# In[3]:


# lista vacía
[]


# Puedes incluir variables y expresiones en la definición de la lista. Python evaluará dichas expresiones y luego construirá la lista. 

# In[4]:


x = "foo"
[2 < 3, x.capitalize(), 5**2, [1, 2]]


# El tipo `list` puede ser utilizado para convertir otros tipos de objetos a listas, en concreto aquellos que sean **iterables**, cosa que definiremos más adelante. 

# In[5]:


list("hello world")


# In[6]:


list(1)


# In[ ]:


list([1])


# ---
# ## Las listas son iterables
# Los **objetos iterables** son aquellos compatibles con operaciones de iteración, la más común de ellas el buque *for*, que veremos más adelante. Otros objetos iterables pueden ser las tuplas, los diccionarios, los conjuntos, los rangos, los generadores, los arrays de numpy etc. Dado un iterable `s`, podemos realizar las siguientes operaciones
# 
# | Operación | Descripción |
# |:---------------|:-----------------|
# | `for vars in s:`| Iteración |
# | `x, y, ... = s`| Deshacer en variables |
# | `x in s` `x not in s`| Pertenencia |
# | `[a, *s, b]`| Expandir |

# Veamos algunos ejemplos

# In[ ]:


items = [3, 4, 5]
x, y, z = items
print(x, y, z)


# In[ ]:


"a" in items


# In[ ]:


a = [1, 2, *items, 6]
a


# Existen muchas funciones predefinidas que aceptan un iterable como argumento, aquí exponemos algunas de ellas
# 
# | Función | Descripción |
# |:---------------|:-----------------|
# | `list(s)`| Crea una lista a partir de `s`|
# | `tuple(s)`| Crea una tupla a partir de `s`|
# | `set(s)`| Crea una conjunto a partir de `s`|
# | `min(s, [,key])`| Mínimo de `s`|
# | `max(s, [,key])`| Máximo de `s`|
# | `any(s)`| Devuelve `True` si algún item de `s` es verdadero|
# | `all(s)`| Devuelve `True` si todos los items de `s` son verdaderos|
# | `sum(s, [, initial])`| Suma de `s` |
# | `sorted(s, [, key])`| Crea una lista ordenada |
# 

# ---
# ## Las listas son secuencias
# 
# Al igual que las cadenas, las listas **son secuencias**, por lo que el orden de sus objetos es importante y podemos acceder a los mismo mediante un índice entero que empieza en cero. 

# In[ ]:


# El orden importa
[1, "a", True] == ["a", 1, True]


# Podemos utilizar la misma sintaxis de indexado que vimos con las cadenas utilizando corchetes y `:`. 
# 
# En las secuencias, el método que nos da la longitud del objeto es `len`

# In[ ]:


x = [2, 4, 6, 8, 10]
len(x)


# In[ ]:


x = "hello world"
len(x)


# In[ ]:


x = 1
len(x)


# ---
# ## Las listas son mutables
# 
# Las listas son utilizadas cuando queramos almacenar datos que pueden cambiar o deben actualizarse, ya que *las listas pueden cambiar una vez han sido creadas*, o dicho de otro modo, son objetos **mutables**. Veámoslo con un ejemplo

# In[ ]:


x = [2, 4, 6, 8, 10]
y = [2, 4, 6, 8, 10]

# asignamos una cadena al segundo objeto de x
x[1] = "apple"
x


# In[ ]:


# podemos realizar asignaciones a nivel de lista
y[1:4] = [-3, -4, -5]
y


# Dos métodos muy utilizados a la hora de manipular listas son el `append` que nos permiten añadir un elemento a la cola de lista y `extend` para añadir varios

# In[ ]:


x = [2, 4, 6, 8, 10]
x.append("foo")
x


# In[ ]:


# a extend tenemos que pasarle una lista de objetos
x.extend([True, False, None])
x


# Estos métodos realizan las operaciones *inplace*, es decir, modifican el objeto sin necesidad de tener que asignarlo. No tiene sentido escribir algo como `x = x.append("foo")`, de hecho el método `append` devuelve `None`

# Otros métodos que pueden ser útiles con listas son 
# - `pop(n)`: devuelve el elemento `n`-ésimo y lo borra de la lista. Devuelve un `IndexError` si `n` está fuera del índice.
# - `remove(a)`: borra el primer elemento cuyo valor coincida con `a`. Devuelve un `ValueError` si no encuentra el valor. 
# - `insert(n, a)`: inserta el objeto `a` en la posición `n`-ésima de la lista

# In[ ]:


x = ["a", "b", "c", "d"]
x.pop(2)


# In[ ]:


x


# In[ ]:


x.append("a")
x


# In[ ]:


x.remove("a")


# In[ ]:


x


# In[ ]:


x.insert(2, None)
x


# Podemos concatenar listas usando el operador `+`. En caso de tener listas anidadas, es posible aplicar directamente más de una operación de indexado. 

# In[ ]:


x = [[1, "foo"], ["bar", None], True, 3.5]
x[1][0]


# :::{exercise}
# :label: lists-index
# 
# Modifica la siguiente lista
# 
# ```
# l = [[1, "foo"], ["bar", None], True, 3.5]
# ```
# para obtener 
# 
# ```
# [[1, "baz"], ["bar", None], True, -1.5, [0, 1, 0]]
# ```
# 
# :::

# :::{exercise}
# :label: lists-comparison
# 
# Condiera una lista `s` y definamos 
# 
# ```
# r = list(s)
# t = s
# ```
# 
# Indica si las siguientes expresiones son verdaderas o falsas 
# - `r == s`
# - `r is s`
# - `s is t`
# 
# :::

# Para el siguiente ejercicio, puedes utilizar el módulo `time` o el [**comando mágico**](https://ipython.readthedocs.io/en/stable/interactive/magics.html) de jupyter `%%time` para obtener el tiempo de ejecución de una celdilla. 

# In[65]:


get_ipython().run_cell_magic('time', '', '\n3**999\n')


# In[70]:


import time

start = time.time()
3**999
end = time.time()
print(f"Execution time: {end - start:0.8f} seconds")


# :::{exercise}
# :label: lists-membership
# 
# Usando la estructura `list(range(n)`, crea listas de diferentes tamaños y estima la complejidad computacional en cuanto a tiempo de cómputo de comprobar la pertenencia a una lista. 
# 
# :::
