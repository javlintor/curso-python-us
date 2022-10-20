#!/usr/bin/env python
# coding: utf-8

# # Principios básicos de Python 
# 
# En esta sección vamos a familiarizarnos con los tipos predefinidos de Python y cómo manejarlos para transformar los diferentes datos de nuestro programa. 
# 
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

# ## Variables
# 
# Una **variable** es un nombre al que asignamos algún valor. Para definir una variable en Python escribimos el nombre de la misma seguido del símbolo igual `=` más el valor que queremos asignarle

# In[11]:


x = 42


# Python tiene **tipado dinánimo**, es decir, no es necesario indicar al intérprete cuál es el tipo de cada variable, si no que lo infiere por sí solo en función de la forma en la que lo escribamos y la expresión en la que se utilice. 

# Para obtener el tipo de un objeto utilizamos la función `type`

# In[12]:


type(42)


# Es posible escribir definiciones en las que explícitamente indiquemos el tipo con un fin meramente informativo

# In[13]:


x: int = 42


# El intérprete de Python ignorará por completo este tipado manual. Esta sintaxis puede ser útil a la hora de mejorar la legibilidad del código y también puede ser utilizado por herramientas de terceros que verifiquen la consistencia del mismo, pero no cambiaremos el tipo de la variable. Por ejemplo

# In[14]:


x: str = 42
type(x)


# In[ ]:





# En realidad una variable es un nombre simbólico que hace de referencia o puntero a un objeto que se crea en memoria. Una vez hemos asignado el objeto a nuestra variable, podemos referenciarla por su nombre. 
# 
# Por ejemplo, al crear la variable 
# ```
# n = 300
# ```
# estamos creando un objeto de tipo `int` con el valor 300 asignado a la variable `n`. 

# ![variable.webp](data:image/webp;base64,UklGRq4FAABXRUJQVlA4TKIFAAAve0IuEA8GKbJt18r6nzRDCk7wb4F5Tm83SYSDtpEkyayO1CE+JG9XIDWSJEmy1PyGxSBZ/kiWwxzhbmzbdpJNButcO1CXs9IBxSr7yVCHkKKH128SFUVYhAUqQlREBSyiIkSyUdllN0IRlY2KAEtlF9il8vUbURHH70ds2aWy2H8UsctuhEWx7CK2VMRujn8qi8/31z/bIipbLFssx382KqLy+Z3d/FOKLYpll92IXWD/ETZCVCxbNirHpbJFuBGK2KWyRVi2WHZRXD3iSCJEH3OM0ccUQ8zRxhRNhOhjOJFFEZcIEbgplrFAH8t0PhFHlIaUcnWE3JBrI8TzTJNr43EisO8XuxOLVMTirPjk9rk0isOduHDjm7Lx5Pl98eg//MZH7/He/3j7/BvZ8e7xOnAUpG3ArP6F7xmIiAnQK+ARaUKwbUeRMs3WZzLu7u4uuLM1KANScyqFJKL/btg2ciROsb6cqk939/+dbAmADEmKyB7frW3btm3bts2ZqYj88evqzLpcPEf034HbSIqUpePLMnX3A0K/Our1nBcFvZzv+r/d27ktaro4r8X+Y/+x/9h/7D/2H/uP/cf+Y/85Ak/h0NEjIn6cjRR6zPHXLN0rVPdLNX4QcfMlEsaDSDXPbT7D1aLzjVB76/SAiJsP4ev6QK45TDMhfpNRW4kAETf/IX4TDZqHfJ4I3m4zAkTcfIeITTRqzlPYnfiUgdqwBSJunkOLMgMXPAz1yoBgGYCEm+cwixrhlYoIAFdCzb7SeC+iSb5Ajf0SL3hL9h0S0SDiRjj39jjMAEdjNern2qM03WijhvQTb0+f0jRaQMSNcNq9e/yqgv/YiqJErB6N3T8Ucafc/t+zoHmcxoGIG90cId67x86qvQ8vxNMmqXc21wkUdUo76c4kRDIftI1pHYi4Ea53ET6Cba/lJVyzuJfQZsCiVpRhzWTJLeNGN0cAIdgl64kl8crmKYt0K3q3uUqUcSPcZEWQAn78dnI7QFD8qYuQRB7YhEpl3L4YORo52MLVHtToA4JmkxNumNZcmmXcsAq2g0oUIFYYbAOXbVTBBt6KfUR+r92t0Q0ABi4ybpgEsygIlVtflyIgV2fnso0yYBa/EEayGbD4S4/VDNUaszJumFyGQZTtxkBZ4fGzuzaagJZvmY8+4KlkqNT5Xzd8GmBckXHgro0uIJXzQrMTD78sEF24a6MP+IU6XN8ElB3mVX1018YgIBRYR7JRM/7vBLJRVBKj1VMoWE+T4rHYf4EhOFoZBYSjzAeRbKY8P5XP1zonqcl7/iYvrMfEnQFkbjHOyIrJaj1Ujodd1u9GZPSkWJWToOHOYv17tM9pW7mnjnD9zF8Lg8fTc5Gr9jZKqF/Li1hErRn8DtcVEcutQ3qtB0edTJ++cTLhzJGTmG9zP4V0jvpkwfkQSt5tXvO5e8TTDKOO+8gibqRT1CcLBLa5g8bSkM1BpIFaqBRE3CjnqEm2nAdmliPxyiBrKevJETco4kY5RU2ylHuJT/K1Bpzx0iJuhHNUJ9wM0o0+i79NngjhFFXJHq5e1HfSfO7bss8kP07CjXCOioRW+aXay7ZooieUxpybL5qg+vQbfL4hRVy1kXAjnKKU8Mp7QkS1X58SoX0/NqtpwxG6+l5Crwf7RrUOLpIBJNwI5ygmzLrRvJl0s5nrfKPqAUDCjXDaHwm3DnOOU4DGPanBR9hIuCFzVwZFle8Em7piTVIHap0HJvPBlIOkX6nu3zINykov3z001c/4z6l1/quCSyPCY38NBKPWGNaiAETckLkCTV0JbyPUrAm5zr3ma+AfBGfTjUvyMjWcZdy8BfqSYM340W3Y5uNmsz//v+rbrxL17cXcrOvwDdf1sGwUv3dNGfYf+4/9x/5j/7H/2H/sP/Yf++8vB7/3/4X5vf9t7S+vEA==)

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

