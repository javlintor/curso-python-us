#!/usr/bin/env python
# coding: utf-8

# # Capítulo 1 
# 
# 1. Bienvenida y presentación 
# 2. Qué es Python y por qué puede ser una herramienta útil
# 3. Python como lenguage de programación 
#     - Pros y Cons 
#         - Pros: Simple (Comparación con otros lenguajes), Amigable, Ampliamente utilizado (empresas que lo utilizan: desde servicios web, DS, Big Data), comunidad enorme (gráfica SOF, salario etc), 
#         librerías open source altamente optimizadas. 
#         Tiene implementadas 
#         - Cons: quizás, dada una tarea, no sea la herramienta más óptima para llevarla acabo 
# 4. Autor y año: nombre. 
#     - Python started as a hobby project by Guido Van Rossum and was first released in 1991
# 5. Python es un lenguaje interpretado (a diferencia de otros compilados, explicar diferentes)
# 6. Definición técnica 
# 7. Versión actual de Python (explicar cómo se versiona el software)
# 8. Dónde ejecutar Python (Ejecutar el intérprete desde la terminal, ipython, cuaderno de python .ipynb - tanto desde jupyter como desde VS)
# 9. Instalar python. Distribuciones de python. 
# 10. Pip como instalador de paquetes

# ## Definición técnica de qué es Python: 
# 
# > Python is an *interpreted*, *high-level*, *general-purpose* programming language. It is is *dynamically typed* and *garbage-collected*.

# ## Python como lenguaje interpretado
# 
# Los programas de Python son ejecutados por un **intérprete**, que puede estar implementado en otros lenguajes como Java o C. Esto supone un paso intermedio entre el código entendible por los humanos y el código máquina, que es el *bytecode*, una serie de instrucciones de bajo nivel que sí pueden ser ejecutadas por in intérprete. 
# 
# Cuando durante la ejecución de un programa importes un módulo, en la misma carpeta del script se crea otra llamada `__pycache__`, que contiene archivos del tipo `.pyc` o `.pyo` que contienen este *código byte*, por ejemplo 
# ```
# /folder   
#     - __pycache__       
#         - preprocess.cpython-36.pyc   
#     - preprocess.py
# ```
# La extensión `.cpython-36` hace referencia al tipo de intérprete utilizado, en este caso implementado en C. Como programadores podemos ignorar estos archivos, sólo tener en cuenta que están ahí para hacer nuestros programas un poco más rápidos. 

# ## Dónde ejecutar Python
# 
# Hay muchos entornos en los que un intérprete de Python puede correr como un IDE (Integrated Development Environment -como Visual Studio Code, Pycharm o Atom-), un navegador o una terminal. 
