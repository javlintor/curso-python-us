#!/usr/bin/env python
# coding: utf-8

# # Principios básicos de Python 
# 
# En esta sección vamos a familiarizarnos con los tipos predefinidos de Python y cómo manejarlos para transformar los diferentes datos de nuestro programa. 
# 
# ---
# ## Predefinidos, Variables y Expresiones
# 
# Python incorpora una serie de tipos predefinidos o primitivos como **enteros**, **flotantes**, **cadenas** o **booleanos**

# ```
# 42              # int 
# 4.2             # float
# "forty-two"     # str
# True            # bool
# complex(1, -1)  # complex
# ```

# ---
# ## Variables
# 
# Una **variable** es un nombre al que asignamos algún valor. Para definir una variable en Python escribimos el nombre de la misma seguido del símbolo igual `=` más el valor que queremos asignarle

# In[1]:


x = 42


# Python tiene **tipado dinámico**, es decir, no es necesario indicar al intérprete cuál es el tipo de cada variable, si no que lo infiere por sí solo en función de la forma en la que lo escribamos y la expresión en la que se utilice. 

# Para obtener el tipo de un objeto utilizamos la función `type`

# In[2]:


type(42)


# Es posible escribir definiciones en las que explícitamente indiquemos el tipo con un fin meramente informativo

# In[3]:


x: int = 42


# El intérprete de Python ignorará por completo este tipado manual. Esta sintaxis puede ser útil a la hora de mejorar la legibilidad del código y también puede ser utilizado por herramientas de terceros que verifiquen la consistencia del mismo, pero no cambiaremos el tipo de la variable. Por ejemplo

# In[4]:


x: str = 42
type(x)


# En realidad una variable es un nombre simbólico que hace de referencia o puntero a un objeto que se crea en memoria. Una vez hemos asignado el objeto a nuestra variable, podemos referenciarla por su nombre. 
# 
# Por ejemplo, al crear la variable 
# ```
# n = 300
# ```
# estamos creando un objeto de tipo `int` con el valor 300 asignado a la variable `n`. 

# ![picture](https://drive.google.com/uc?id=1LxoXumDnVh9kwvcSFYoN_1mIG39qO5LV)
# 

# Podemos consultar la dirección del objeto creado en memoria con la función `id` y el número de bytes reservado para guardarlo con la `getsizeof` dentro del módulo `sys`.

# In[6]:


import sys

n = 300
print(id(n))
print(sys.getsizeof(n))


# Si ahora realizamos otra asignación a la variable ya creada

# In[7]:


m = n


# Python no creará un nuevo objeto, si no solamente una nueva referencia, que apunta al mismo objeto creado con anterioridad

# In[8]:


print(id(m))


# ![picture](https://drive.google.com/uc?id=1wI1R_M0z9A8k5ynOnlO8U3FK10wUKZwf)
# 

# Asignando la variable `m` a otro objeto, su referencia en memoria cambiará

# In[9]:


m = 400
print(id(m))


# ![picture](https://drive.google.com/uc?id=1L6hU-KtS7T9nZNtx_hc2650Zp5Er3WJ5)

# En caso de que `n` cambie también su referencia, el objeto que guardaba el entero `300` se queda sin referencias

# In[10]:


n = "foo"


# ![picture](https://drive.google.com/uc?id=1535KCfXgHMpIOSyBn67llPnBTeLNszf3)

# El ciclo de vida de un objeto comienza cuando se crea, y durante el mismo puede tener múltiples referencias que se van creando y borrando. Sin embargo, cuando el número de referencias es cero, el objeto queda inaccesible. 
# 
# Python incorpora lo que se denomina un [**recolector de basura**](https://en.wikipedia.org/wiki/Garbage_collection_(computer_science)), o *garbage collector*, que se encarga de reclamar la memoria que ocupan los objetos que son innaccesibles. 
# 
# Finalmente, es importante notar que a la hora de nombrar las variables podemos utilizar cadenas de longitud arbitraria que sean combinación de letras mayúsculas y minúsculas, dígitos y guión bajo (`_`) -siempre que no empiecen por un dígito-. No obstante, según la guía de estilos para Python [PEP 8](https://peps.python.org/pep-0008/#function-and-variable-names) los nombres de variables deben ser en minúscula y con espacios separados por guiones bajos, lo que se denomina *Snake Case*.

# In[11]:


numberofcollegegraduates = 2500        # mal
NUMBEROFCOLLEGEGRADUATES = 2500        # uppercase, reservado para variables globales
numberOfCollegeGraduates = 2500        # Camel Case, reservado para clases
NumberOfCollegeGraduates = 2500        # mal
number_of_college_graduates = 2500     # Snake Case     


# :::{exercise}
# :label: my-exercise
# 
# Crea objetos de tipo `int`, `float`, `complex`, `str` y consulta cuánta memoria ocupan. ¿Depende el espacio sólo del tipo? 
# 
# :::

# ---
# ## Expresiones
# 
# Una **expresión** es una combinación de valores, variables y operadores que produce un resultado. 
# 
# Escribiendo expresiones podemos utilizar Python como si de una calculadora se tratara, utilizando las operaciones aritméticas habituales (`+`, `-`, `*`, `/`) con el orden habitual de estas.
# 
# ```{code} 
# 2 + 2            # 4
# 50 - 5*6         # 20
# (50 - 5*6) / 4   # 5.0
# 8 / 5            # 1.6
# ```

# In[ ]:


2 + 3 * 4

