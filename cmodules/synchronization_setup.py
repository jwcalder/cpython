#python3 synchronization_setup.py build_ext --inplace
from distutils.core import setup, Extension
import numpy

# define the extension module
synchronization = Extension('synchronization', sources=['synchronization.c','memory_allocation.c'],include_dirs=[numpy.get_include()],extra_compile_args = ['-Ofast'])

# run the setup
setup(ext_modules=[synchronization])
