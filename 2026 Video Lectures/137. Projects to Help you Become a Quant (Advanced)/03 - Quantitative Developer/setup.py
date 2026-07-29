"""Build script for the Cython Monte Carlo pricer.

Usage
-----
    python setup.py build_ext --inplace

This compiles ``mc_cython.pyx`` into a native extension module
(``mc_cython.<abi>.pyd`` on Windows / ``.so`` on Linux/macOS) placed next to the
source files so ``import mc_cython`` works from this directory.

Requirements: a working C compiler (MSVC "Build Tools for Visual Studio" on
Windows, gcc/clang on Linux/macOS), plus ``cython`` and ``numpy`` installed.
If no compiler is available the build will fail loudly here -- but the rest of
the project (``benchmark.py``) still runs fine without the compiled extension.
"""

from setuptools import Extension, setup

import numpy as np
from Cython.Build import cythonize

extensions = [
    Extension(
        name="mc_cython",
        sources=["mc_cython.pyx"],
        include_dirs=[np.get_include()],
        # Silence the deprecated-NumPy-API warning from `cimport numpy`.
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

setup(
    name="mc_cython",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    ),
    zip_safe=False,
)
