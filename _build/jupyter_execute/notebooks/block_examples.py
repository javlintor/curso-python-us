#!/usr/bin/env python
# coding: utf-8

# # Bloques de ejemplo 
# 
# A lo largo del curso utilizaremos direntes bloques o entornos para esquematizar el contenido. A continuación vamos a describirlos
# 
# ## Índice y bibliografía 
# Para acceder al índice y la bilbiografía, accedemos mediante los bloques `{tableofcontents}` y `{bibliography}`.
# 
# ## Entornos predefinidos 
# Veamos los entornos que tenemos disponibles en **Jupyter Book**
# 
# ### Nota 
# :::{note}
# Esto es una nota
# :::
# 
# ### Aviso 
# :::{warning}
# Esto es un aviso 
# :::
# 
# ### Véase
# :::{seealso}
# Esto es un véase
# :::
# 
# ### Pista
# :::{tip}
# Esto es una pista
# :::
# 
# ### Bloques customizados 
# Podemos añadir bloques customizados mediante `{admonition}`
# 
# ```{admonition} Mi bloque customizado
# Foo, Bar and Baz!
# ```
# 
# ```{admonition} Pista
# :class: tip
# Esto es una pista
# ```
# 
# ### Bloque de código 
# :::{code}
# def block_code(a, b):
#     return a + b
# :::
# 
# 
# ### Bloque de código python (parece que es el que viene por defecto)
# :::{code} python3
# import numpy as np
# 
# np.sqrt(4)
# :::
# 
# ### Bloque de código R 
# :::{code} R
# library(data.table)
# 
# dt <- data.table(a = c(1, 2, 3), b = c(4, 5, 6))
# :::
# 
# ### Estructura de carpetas 
# ```console
# dir/
#     - file1.txt
#     - file2.txt
#     - subdir/
#         - file3.txt
# ```
# 
# ### Látex
# Podemos incluir ecuaciones en línea con un solo dólar $x^n + y^n = z^n$. O también podemos escribir ecuaciones en bloque finalizamos la sección.
# 
# ### Secciones 
# 
# Para crear una etiqueta en una sección, escribimos el nombre de la etiqueta entre parántesis e igualamos antes de comenzar el contenido correspondiente. 
# 
# Por ejemplo, escribiendo 
# 
# ```
# (my-section)=
# ## My section
# bla bla bla ...
# ```
# 
# podemos etiquetar una sección. Para referenciarla, podemos utilizar `[texto](nombre-de-la-etiqueta)`. Por ejemplo, es la sección anterior en la que hablamos de látex. 
# 
# 
# Con ecuaciones es un poco diferente, escribimos la etiqueta entre parátesis justo después de cerrar el segundo dólar.
# 
# $$
# f(x) = \frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{1}{2}(x - \mu)^2} 
# $$ (my-equation)
# 
# y ahora la puedo referenciar ``` {eq}`my-equation` ```, como aquí {eq}`my-equation`
# 
# ## Ejercicios 
# 
# Para definir un ejercicio utilizamos `{exercise}`, con su correspondiente etiqueta para referenciarlo mediante `:label:`. Por ejemplo, considera el siguiente ejercicio
# ```{exercise}
# :label: my-exercise
# 
# Recall that $n!$ is read as "$n$ factorial" and defined as
# $n! = n \times (n - 1) \times \cdots \times 2 \times 1$.
# 
# There are functions to compute this in various modules, but let's
# write our own version as an exercise.
# 
# In particular, write a function `factorial` such that `factorial(n)` returns $n!$
# for any positive integer $n$.
# ```
# 
# Las soluciones vienen con bloques `:::{solution}`, seguidas de la etiqueta del ejercicio, también con su correspondiente `:label:` si se quiere referenciar y `:class: dropdown` si queremos ocultar la solución.
# 
# :::{solution} my-exercise
# :label: my-solution
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
# 
# Podemos referenciar al ejercicio, {ref}`my-exercise` y también a la solución {ref}`my-solution`
# 
# ### Citas a la bibliografía 
# 
# Para escribir una cita utilizamos ``` {cite}`perez2011python` ```, como en este caso {cite}`perez2011python`.

# 
