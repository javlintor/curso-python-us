#!/usr/bin/env python
# coding: utf-8

# # Gestión de proyectos en Python 

# ## Gestión de paquetes - Poetry  
# 
# [Poetry](https://python-poetry.org/) es un sistema de gestión de proyectos en Python que va más allá de manejar las dependencias de un proyecto como podemos hacer con [pipenv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)  
# Algunos lenguanjes como go o rust tienen utilidades de líneas de comandos integrados que nos permiten manejar detalles sobre nuestro proyecto 
# 
# - configuración e instalaciones 
# - manejo de dependencias 
# - compilación y ejecuciones 
# 
# Python no tiene una utilidad nativa pero sistemas desarrollados por terceros nos pueden ayudar a realizar estas tareas de forma precisa, como es el caso de poetry. 
# 
# ### Instalación 
#  
# No se instala como un paquete de Python, si no como una aplicación independiente. Por ejemplo, podemos ejecutar el siguiente comando 
# 
# ```
# curl -sSL http://install.python.python-poetry.org | python3 -
# ```
# 
# aunque los detalles de instalación pueden cambiar en función de cómo tengamos configurado nuestro sistema. Para verificar que hemos instalado correctamente poetry podemos ejecutar 
# 
# ```
# poetry --version 
# ```
# 
# y obtener un output como el siguiente `Poetry (version 1.3.1)`. 
# 
# ### Creando un nuevo proyecto 
# 
# Para crear un proyecto con poetry (podemos pensar que vamos a desarrollar un paquete de python) utilizamos el siguiente comando 
# 
# ```
# poetry new py4d
# ```
# 
# Esto debería de crear una carpeta py4d con la siguiente estructura 
# 
# - Un archivo README.md como descripción del paquete 
# - Otro archivo pyproject.toml de configuración, que no debemos tocar directamente
# - Una carpeta de tests para incluir nuestros tests unitarios
# - Otra carpeta py4d donde estarán los ficheros con las funciones y clases de nuestro paquete. 
# 
# ```
# py4d
# ├── pyproject.toml
# ├── README.md
# ├── py4d
# │   └── __init__.py
# └── tests
#     └── __init__.py
# 
# ```
# 
# Por defecto, un nuevo proyecto no tiene un entorno virtual asociado por defecto, si no que Poetry utiliza en entorno virtual que crea en 
# 
# - macOS: `~/Library/Caches/pypoetry`
# - Windows: `C:\Users\<username>\AppData\Local\pypoetry\Cache`
# - Unix: `~/.cache/pypoetry`
# 
# Así que en principio no tenemos que crear nada más para empezar a añadir dependencias. Sólo tenemos que modificar el `pyproject.toml`, en concreto la sección `tool.petry.dependencies` o más fácil, añadirlas con la CLI de poetry
# 
# ```
# poetry add pandas
# ```
# 
# Poetry tiene [una sintaxis de expecificación de dependencias](https://python-poetry.org/docs/dependency-specification/) muy rica, como por ejemplo si queremos replicar justo la versión que hay en google colab de pandas,
# 
# ```
# poetry add pandas==1.3.5
# ```
# 
# Podemos lanzar un script con poetry simplemente escribiendo `poetry run python your_script.py`. Del mismo modo si utilizas herramientas desde línea de comandos puedes lanzarlas con poetry, como `poetry run black .`
# 

# ## Limpia tu código - flake8 + black
# 
# 
# ### Black 
# 
# Es una herramienta de **formateo** de código, es decir, al aplicar black a nuestro código, va a modificar nuestro código según para que siga ciertas guías de estilo, normalmente basadas en [PEP8](https://peps.python.org/pep-0008/), que son configurables por nosotros mismos, de modo que se cambiará el estilo del código pero nunca se modificará el output del intérprete de Python. 
# 
# [Black](https://black.readthedocs.io/en/stable/) tiene un principio básico y es que todo el código que formatea debe ser similar, haciendo que estilo pueda cambiar bastante del de partida, pero minimizando los cambios respecto a las diferentes versiones del proyecto u otros proyectos igualmente formateados. 
# 
# Podemos instalar black utilizando `pip`, siempre que tengamos Python 3.7 o superior. 
# 
# ```
# pip install black
# ```
# 
# Si queremos formatear nuestros notebooks de jupyter, debemos instalarlo con 
# 
# ```
# pip install black[jupyter]
# ```
# 
# Se emplea de forma muy sencilla, simplemente llamando al comando `black` en una terminal 
# 
# ```
# black {fichero_que_queremos_formatear.py}
# ```
# 
