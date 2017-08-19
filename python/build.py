from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(
        ['games/omringa/game_state.pyx'],
        compiler_directives={
            'boundscheck': False,
            'wraparound': False,
            'initializedcheck': False,
            'nonecheck': False,
            'overflowcheck': False,
            'cdivision': True,
            'infer_types': True,
            'optimize.use_switch': True,
            'optimize.unpack_method_calls': True,
        },
        language='c++'),
)

# python build.py build_ext --inplace
