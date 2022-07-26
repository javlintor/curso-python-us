#!/usr/bin/env python
# coding: utf-8

# # Plantilla
# 
# Es esta introducción voy a ir mostrando los diferentes tipos de bloques que podemos ir utilizando.
# 
# ## Curso de Python
# 
# Este libro contiene un curso de introducción al lenguaje de programación Python orientado a estudiantes de Estadística y/o Matemáticas 
# 
# Puedes consultar [the Jupyter Book documentation](https://jupyterbook.org) para informarte sobre cómo hemos diseñado este libro
# 
# ### Diferentes tipos de bloques
# Puedes añadir bloque de códigos como el siguiente 
# 
# ```python
# def f(a, b):
#     return a + b
# ```
# Puedes añadir notas 
# 
# :::{note}
# This is a note
# :::
# 
# También avisos 
# 
# :::{warning}
# This is a warning
# :::
# 
# :::{seealso}
# This is a see also
# :::
# 
# (intro-mas-detalles)=
# ## Más detalles 
# Por otro lado puedes añadir ecuaciones en línea $ x^n + y^n = z^n $ o $x \in \mathbb{R}$.
# 
# Si quieres añadir bloques de ecuaciones, lo hacemos con doble dolar
# 
# $$
#   \int_0^\infty \frac{x^3}{e^x-1}\,dx = \frac{\pi^4}{15}
# $$ (integral)
# 
# o también puedes incluirlo con el indicardor `\begin{equation}`
# 
# \begin{gather*}
# a_1=b_1+c_1\\
# a_2=b_2+c_2-d_2+e_2
# \end{gather*} (sistem-of-equations)
# 
# \begin{align}
# a_{11}& =b_{11}&
#   a_{12}& =b_{12}\\
# a_{21}& =b_{21}&
#   a_{22}& =b_{22}+c_{22}
# \end{align}
# 
# De este modo también puedes hacer referencia a equaciones, como la integral {eq}`integral`
# 
# Here is my nifty citation {cite}`perez2011python`.

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
# :::{solution} my-exercise
# :label: my-solution
# :class: dropdown
# 
# Here's one solution.
# 
# ```{code-block} python
# def factorial(n):
#     k = 1
#     for i in range(n):
#         k = k * (i + 1)
#     return k
# 
# factorial(4)
# ```
# :::

# ```{exercise}
# :label: my-exercise-2
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
# :::{solution} my-exercise-2
# :label: my-solution-2
# :class: dropdown
# 
# Here's one solution.
# 
# ```{code-block} python
# def factorial(n):
#     k = 1
#     for i in range(n):
#         k = k * (i + 1)
#     return k
# 
# factorial(4)
# ```
# :::

# 
