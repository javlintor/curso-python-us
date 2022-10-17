#!/usr/bin/env python
# coding: utf-8

# # Principios básicos de Python 
# 
# En esta sección vamos a familiarizarnos con los tipos predefinidos de Python y cómo manejarlos para transformar los diferentes datos de nuestro programa. 
# 
# ## Predefinidos, Variables y Expresiones
# 
# Python incorpora una serie de tipos predefinidos o primitivos como **enteros**, **flotantes**, **cadenas** o **booleanos**

# In[ ]:


42              # int 
4.2             # float
"forty-two"     # str
True            # bool


# ## Variables
# Una **variable** es un nombre al que asignamos algún valor. Para definir una variable en Python escribimos el nombre de la misma seguido del símbolo igual `=` más el valor que queremos asignarle

# In[1]:


x = 42


# Python tiene **tipado dinánimo**, es decir, no es necesario indicar al intérprete cuál es el tipo de cada variable, si no que lo infiere por sí solo en función de la forma en la que lo escribamos y la expresión en la que se utilice. 

# Para obtener el tipo de un objeto utilizamos la función `type`

# In[ ]:


type(42)


# Es posible escribir definiciones en las que explícitamente indiquemos el tipo con un fin meramente informativo

# In[ ]:


x: int = 42


# El intérprete de Python ignorará por completo este tipado manual. Esta sintaxis puede ser útil a la hora de mejorar la legibilidad del código y también puede ser utilizado por herramientas de terceros que verifiquen la consistencia del mismo, pero no cambiaremos el tipo de la variable. Por ejemplo

# In[ ]:


x: str = 42
type(x)


# ## Expresiones
# Una **expresión** es una combinación de valores, variables y operadores que produce un resultado. 
# 
# Escribiendo expresiones podemos utilizar Python como si de una calculadora se tratara.
# 
# ```{code} 
# 2 + 2            # 4
# 50 - 5*6         # 20
# (50 - 5*6) / 4   # 5.0
# 8 / 5            # 1.6
# ```

# In[2]:


2 + 3 * 4

