from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(['games/omringa/game_state.pyx']),
)

# python build.py build_ext --inplace
