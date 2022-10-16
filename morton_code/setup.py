from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
  name = 'morton_code',
  ext_modules = cythonize("core.pyx"),
  include_dirs=[numpy.get_include()]
)
