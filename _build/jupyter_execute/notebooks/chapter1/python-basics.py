#!/usr/bin/env python
# coding: utf-8

# # Principios básicos de Python 
# 
# En esta sección vamos a familiarizarnos con los tipos predefinidos de Python y cómo manejarlos para transformar los diferentes datos de nuestro programa. 
# 
# ## Predefinidos, Variables y Expresiones
# 
# Python incorpora una serie de tipos predefinidos o primitivos como **enteros**, **flotantes**, **cadenas** o **booleanos**

# In[1]:


42              # int 
4.2             # float
"forty-two"     # str
True            # bool


# :::{admonition} Variables
# :class: seealso
# Una **variable** es un nombre al que asignamos algún valor. Para definir una variable en Python escribimos el nombre de la misma seguido del símbolo igual `=` más el valor que queremos asignarle
# :::

# In[2]:


x = 42


# :::{note} 
# Python tiene **tipado dinánimo**, es decir, no es necesario indicar al intérprete cuál es el tipo de cada variable, si no que lo infiere por sí solo en función de la forma en la que lo escribamos y la expresión en la que se utilice. 
# :::

# Para obtener el tipo de un objeto utilizamos la función `type`

# In[3]:


type(42)


# Es posible escribir definiciones en las que explícitamente indiquemos el tipo con un fin meramente informativo

# In[4]:


x: int = 42


# El intérprete de Python ignorará por completo este tipado manual. Esta sintaxis puede ser útil a la hora de mejorar la legibilidad del código y también puede ser utilizado por herramientas que verifiquen la consistencia del mismo, pero no cambiaremos el tipo de la variable. Por ejemplo

# In[5]:


x: str = 42
type(x)


# :::{admonition} Expresiones
# :class: seealso
# Una **expresión** es una combinación de valores, variables y operadores que produce un resultado. 
# :::
# 
# Escribiendo expresiones podemos utilizar Python como si de una calculadora se tratara.
# 
# ```{code} 
# 2 + 2            # 4
# 50 - 5*6         # 20
# (50 - 5*6) / 4   # 5.0
# 8 / 5            # 1.6
# ```

# In[6]:


2 + 3 * 4


# # Classes
# :::{exercise}
# :label: stack-class
# 
# Construye la clases `Stack` o pila para crear una estructura tipo pila para almacenar datos. Esta clase hereda de `List`, pero además incluye dos métodos nuevos 
# 
# :::

# :::{solution} stack-class
# :label: stack-class-solution
# :class: dropdown
# 
# Here's one solution.
# 
# ```{code}
# def factorial(n):
#     k = 1
#     for i in range(n):
#         k = k * (i + 1)
#     return k
# 
# factorial(4)
# ```
# 
# :::

# In[ ]:





# In[ ]:





#  
