#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/javlintor/curso-python-us/blob/main/notebooks/introduction-python/numeric-types.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Tipos numéricos
# 
# El conjunto de operaciones entre tipos numéricos se resumen en la siguiente tabla 
# 
# | Operation |                       Description                       |
# |:---------:|:-------------------------------------------------------:|
# | `x + y  `   | Suma de dos números                                     |
# | `x - y  `   | Reta de dos números                              |
# | `x * y  `   | Producto de dos números                                 |
# | `x / y  `   | Cociente de dividir dos números                                |
# | `x // y `   | Cociente entero de dividir dos números        |
# | `x % y  `   | Resto de dividir `x` entre `y` |
# | `x ** y `   | `x` elevado a `y`                                 |
# | `-x     `   | Cambiar el signo de `x`                                        |
# | `abs(x)   ` | Valor absoluto                          |
# | `x == y `   | Comprobar si dos números tienen el mismo valor                |
# | `x != y `   | Negación de `==`              |
# | `x > y  `   | Verifica si `x` es estrictamente mayor que `y`                            |
# | `x >= y `   | Verifica si `x` es mayor o igual que `y`                |
# | `x < y  `   | Verifica si `x` es estrictamente menos que `y`                               |
# | `x <= y `   | Verifica si `x` es menor o igual que `y`                 |

# ---
# ## Enteros
# 
# Como hemos visto ya, los **enteros** en Python son utilizados para representar el conjunto de números enteros $\mathbb{Z} = \{\dots, -2, -1, 0, 1, 2, \dots \}$ mediante el tipo predefinido `int`. 
# 
# Puedes crear enteros arbitrariamente grandes, Python reservará tanta memoria como sea necesario (hasta quedarnos sin memoria) para almacenarlo.

# La conversión de `float` a `int` se realiza truncando los decimales mediante la función `math.floor`.

# In[1]:


a = 1.3
print(int(a))


# In[2]:


b = 1.7
print(int(b))


# ---
# ## Flotantes

# Los **flotantes** se utilizan para representar números reales con un número determinado de cifras decimales. El tipo para representar estos objetos es `float`

# In[3]:


isinstance(1, float)


# In[4]:


isinstance(1., float)


# In[5]:


type(10 / 5)


# In[6]:


float("0.43784")


# In[7]:


float(-3)


# Los flotantes también se pueden definir utilizando **notación científica**, que se implementa mediente el carácter `e` para simbolizar $\times 10$. Por ejemplo
# - $2.5 \times 10^3 →$ `2.5e3` 
# - $1.34 \times 10^{-7} →$ `1.34e-7` 

# In[8]:


a = 2.5e3
b = 1.34e-7
print(a)
print(b)


# En este último ejemplo, vemos que a la hora de llamar a la función `print`, Python decide si mostrar el número en notación científica o escribir todas las cifras. 

# Aunque los enteros de Python pueden tener una longitud tan grande como queramos, los flotantes *tienen una cantida limitada de decimales que pueden almacenar*, en concreto Python dispone de 64 bits de memoria en la mayoría de los casos para guardar el número. Esto se traduce en que usualmente tendremos una capacidad máxima de almacenar **16 decimales** cuando el número se escribe en notación científica. 
# 
# En el siguiente ejemplo vemos que al convertir un entero de 100 dígitos a flotante sólo podemos retener 16. 

# In[9]:


# Creamos una cadena de longitud 100 llena de 1s
digit_str = "1"*100 
# Convertirmos a entero
digit_int = int("1"*100)
# Convertimos a float
digit_float = float(digit_int)


# In[10]:


digit_int


# In[11]:


digit_float


# Modificar un flotante más allá de su precisión no causará ningún efecto

# In[12]:


digit_float == digit_float + 1


# Muchas veces el hecho de tener una precesición numérica finita puede llevar a comportamientos inesperados

# In[13]:


0.1 + 0.1 + 0.1 - 0.3 == 0


# In[14]:


a = 0.1 + 0.1 + 0.1 - 0.3
print(a)


# > Nunca se debería comprobar directamente si un flotante es igual a otro, en su lugar debería de comprobarse si dos flotantes están arbitrariamente cerca

# Para ello contamos con la función `isclose` del módulo `math`. 

# In[15]:


import math
math.isclose(a, 0, abs_tol=1e-5)


# ---
# ## Operadoes de asignación aumentada 
# 
# En Python disponemos de un ajato muy útil a la hora de actualizar una variable vía una operación aritmética. Por ejemplo, si tenemos una variable `x` y queremos aumentar su valor en `1`, en lugar de escribir 

# In[16]:


x = 1
x = x + 1
print(x)


# Podemos utilzar el **operador de asignación aumentada** asociado a `+` 

# In[17]:


x = 1
x += 1
print(x)


# Otros ejemplos pueden ser los operadores `*=`, `-=`, `/=`, `**=`, `%=` etc. En general, lo que estamos haciendo es calcular un nuevo valor de nuestra variable y posteriormente asignándolo a la misma. Esta técnica también se puede utilizar con tipo que no sean numéricos, por ejemplo con una cadena 

# In[18]:


a = "foo"
a += "bar"
print(a)


# ---
# ## Otros tipos numéricos 
# 
# Existen módulos en la librería estándar que extienden los tipos predefinidos, como pueden ser el módulo [fractions](https://docs.python.org/3/library/fractions.html#module-fractions) para trabajar con representaciones exactas de los números racionales o [decimal](https://docs.python.org/3/library/decimal.html) para controlar la precisión de los decimales.

# :::{exercise}
# :label: numeric-types-exp-abs
# 
# Usando funciones del módulo `math`, calcula el valor de la función $f(x) = e^{|x - 2|}$ en $x = -0.2$
# 
# :::

# :::{exercise}
# :label: numeric-types-cientific-notation
# 
# Haz uso de la notación científica para calcular el orden de magnitud del inverso del cuadrado de la longitud de Planck.
# 
# :::

# :::{exercise}
# :label: numeric-types-decimals
# 
# Con ayuda del módulo `decimals` calcula el décimal número 100 del número $\frac{77}{314}$.
# 
# :::
