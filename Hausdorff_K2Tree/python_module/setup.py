from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
from setuptools import setup, Extension
import glob
import os

# Obtener archivos fuente
def get_source_files():
    sources = ["src/python_bindings.cpp"]
    
    # Agregar archivos del proyecto original
    base_path = ".."
    
    # Snapshot sources
    sources.extend(glob.glob(f"{base_path}/Snapshot/*.cpp"))
    
    # Util sources  
    sources.extend(glob.glob(f"{base_path}/Util/*.cpp"))
    
    # IO sources
    sources.extend(glob.glob(f"{base_path}/IO/*.cpp"))
    
    # Logs sources
    sources.extend(glob.glob(f"{base_path}/Logs/*.cpp"))
    
    # Archivos específicos
    sources.extend([
        f"{base_path}/ryu-kamata.cpp",
        f"{base_path}/utils.cpp"
    ])
    
    # Filtrar main.cpp si existe
    sources = [s for s in sources if not s.endswith("main.cpp")]
    
    return sources

# Directorios de inclusión
include_dirs = [
    pybind11.get_include(),
    "../",
    "../libcds/include",
    "../Snapshot", 
    "../Util",
    "../IO",
    "../Logs"
]

# Directorios de librerías
library_dirs = [
    "../libcds/lib"
]

# Librerías a vincular
libraries = ["cds"]

# Definir la extensión
ext_modules = [
    Pybind11Extension(
        "hausdorff_k2tree_core",
        sources=get_source_files(),
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        language='c++',
        cxx_std=14,
    ),
]

setup(
    name="hausdorff-k2tree",
    version="1.0.0",
    author="Tu Nombre",
    author_email="tu@email.com",
    description="Hausdorff distance algorithms using K2-Tree data structures",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.19.0",
        "torch>=1.7.0",
        "torchvision>=0.8.0"
    ],
    packages=["hausdorff_k2tree"],
    package_dir={"hausdorff_k2tree": "hausdorff_k2tree"},
)