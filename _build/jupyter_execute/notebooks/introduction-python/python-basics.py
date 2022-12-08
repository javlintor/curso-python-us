#!/usr/bin/env python
# coding: utf-8

# # Principios básicos de Python 
# 
# En esta sección y las siguientes vamos a familiarizarnos con los tipos predefinidos de Python y cómo manejarlos para transformar los diferentes datos de nuestro programa. 
# 
# Python incorpora una serie de tipos predefinidos o primitivos como **enteros**, **flotantes**, **cadenas** o **booleanos**

# ```
# 42              # int 
# 4.2             # float
# "forty-two"     # str
# True            # bool
# 1 - 1j          # complex
# ```

# ---
# ## Variables
# 
# Una **variable** es un nombre al que asignamos algún valor. Para definir una variable en Python escribimos el nombre de la misma seguido del símbolo igual `=` más el valor que queremos asignarle

# In[ ]:


x = 42


# Python tiene **tipado dinámico**, es decir, no es necesario indicar al intérprete cuál es el tipo de cada variable, si no que lo infiere por sí solo en función de la forma en la que lo escribamos y la expresión en la que se utilice. 

# Para obtener el tipo de un objeto utilizamos la función `type`. Para verificar si una variable es de un tipo tenemos la función `isinstance`

# In[ ]:


type(42)


# In[ ]:


isinstance("foo", str)


# In[ ]:


import math
type(math.pi)


# En esta última celdilla hemos importado el **módulo** estándar `math`, que incorpora un conjunto de funciones de funciones matemáticas como la exponencial, logaritmos o funciones trigonométricas. Antes de acceder a las funciones definidas en un módulo es necesario importarlo.

# Es posible escribir definiciones en las que explícitamente indiquemos el tipo con un fin meramente informativo

# In[ ]:


x: int = 42


# El intérprete de Python ignorará por completo este tipado manual. Esta sintaxis puede ser útil a la hora de mejorar la legibilidad del código y también puede ser utilizado por herramientas de terceros que verifiquen la consistencia del mismo, pero no cambiaremos el tipo de la variable. Por ejemplo

# In[ ]:


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

# In[ ]:


import sys

n = 300
print(id(n))
print(sys.getsizeof(n))


# Si ahora realizamos otra asignación a la variable ya creada

# In[ ]:


m = n


# Python no creará un nuevo objeto, si no solamente una nueva referencia, que apunta al mismo objeto creado con anterioridad

# In[ ]:


print(id(m))


# ![picture](https://drive.google.com/uc?id=1wI1R_M0z9A8k5ynOnlO8U3FK10wUKZwf)
# 

# Para comprobar si dos variables referencian el mismo objeto utlizamos la función `is`.

# In[ ]:


n is m


# Asignando la variable `m` a otro objeto, su referencia en memoria cambiará

# In[ ]:


m = 400
print(id(m))
print(m is n)


# ![picture](https://drive.google.com/uc?id=1L6hU-KtS7T9nZNtx_hc2650Zp5Er3WJ5)

# En caso de que `n` cambie también su referencia, el objeto que guardaba el entero `300` se queda sin referencias

# In[ ]:


n = "foo"


# ![picture](https://drive.google.com/uc?id=1535KCfXgHMpIOSyBn67llPnBTeLNszf3)

# El ciclo de vida de un objeto comienza cuando se crea, y durante el mismo puede tener múltiples referencias que se van creando y borrando. Sin embargo, cuando el número de referencias es cero, el objeto queda inaccesible. 
# 
# Python incorpora lo que se denomina un [**recolector de basura**](https://en.wikipedia.org/wiki/Garbage_collection_(computer_science)), o *garbage collector*, que se encarga de reclamar la memoria que ocupan los objetos que son innaccesibles. 
# 
# Finalmente, es importante notar que a la hora de nombrar las variables podemos utilizar cadenas de longitud arbitraria que sean combinación de letras mayúsculas y minúsculas, dígitos y guión bajo (`_`) -siempre que no empiecen por un dígito-. No obstante, según la guía de estilos para Python [PEP 8](https://peps.python.org/pep-0008/#function-and-variable-names) los nombres de variables deben ser en minúscula y con espacios separados por guiones bajos, lo que se denomina *Snake Case*.

# In[ ]:


numberofcollegegraduates = 2500        # mal
NUMBEROFCOLLEGEGRADUATES = 2500        # uppercase, reservado para variables globales
numberOfCollegeGraduates = 2500        # mal
NumberOfCollegeGraduates = 2500        # Camel Case, reservado para clases
number_of_college_graduates = 2500     # Snake Case     


# Existe un conjunto de **nombres reservados** que no pueden ser utilizados para nombrar una variable
# 
# |          |         |          |        |
# | -------- | ------- | -------- | ------ |
# | False    | def     | if       | raise  |
# | None     | del     | import   | return |
# | True     | elif    | in       | try    |
# | and      | else    | is       | while  |
# | as       | except  | lambda   | with   |
# | assert   | finally | nonlocal | yield  |
# | break    | for     | not      |        |
# | class    | form    | or       |        |
# | continue | global  | pass     |        |

# :::{exercise}
# :label: python-basics-memory-types
# 
# Crea objetos de tipo `int`, `float`, `complex`, `bool`, `str` y consulta cuánta memoria ocupan. ¿Depende el espacio sólo del tipo? 
# 
# :::

# :::{exercise}
# :label: python-basics-reserved-names-error
# 
# ¿Qué tipo de error obtenemos al nombrar una variable con un nombre reservado?
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


# Otras operaciones importantes entre tipos numéricos son el operador potencia `**`, cociente `//` y resto, `%`. 

# In[ ]:


2 ** 8


# In[ ]:


16 // 3


# In[ ]:


23 % 7


# Como hemos comentado antes, Python puede modificar el tipado de una variable cuando aparece en una expresión. Este **tipado implícito** puede causar errores inesperados, pero a cambio podemos simplificar nuestro código. 
# 
# Por ejemplo, al aplicar el operador suma `+` entre un entero y un flotante, Python convertirá el entero en flotante y aplicará en operador posteriormente.

# In[ ]:


a = 1
b = .5
print(type(a))
print(type(b))
print(type(a + b))


# Para realizar un **tipado explícito**, llamamos directamente al tipo que queremos convertir 

# In[ ]:


a = "8"
b = int("8")
print(type(a))
print(type(b))


# ---
# ## Comentarios
# En Python, podemos comentar nuestro código escribiendo el símbolo `#`, de modo que todo lo que escribamos a la derecha de `#` en la misma línea será ignorado por el intérprete. 

# In[ ]:


# Esto es un comentario 
a = 1 + 2 # Esto es un comentario tras una asignación


# Escribir comentarios cuando estemos creando nuestro código es fundamental, sobre todo de cara a que otros (incluyendo nuestro yo del futuro) podamos entenderlo más fácilmente. No obstante, debería ser más importante centrarnos en que nuestro código sea legible que en escribir comentarios para todo. Véase [este blog](https://stackoverflow.blog/2021/12/23/best-practices-for-writing-code-comments/) para buenas prácticas a la hora de escribir comentarios. 
