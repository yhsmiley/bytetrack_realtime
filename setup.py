from setuptools import setup, find_packages, Extension

import numpy as np


ext_modules = [
    Extension(
        "cython_bbox",
        ["bytetrack_realtime/utils/cython_bbox.pyx"],
        extra_compile_args=["-Wno-cpp", "-Wno-unused-function"],
        include_dirs=[np.get_include()],
    )
]

setup(
    name='bytetrack-realtime',
    version='1.0',
    packages=find_packages(),
    ext_modules=ext_modules,
    install_requires=[
        'cython',
        "lap"
    ]
)
