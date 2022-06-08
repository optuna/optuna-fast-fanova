#!/bin/bash

# Please execute the below command to build manylinux wheels.
# $ docker run -it --platform linux/amd64 --rm -v $(pwd):/io quay.io/pypa/manylinux_2_24_x86_64 /io/tools/build_wheel.sh

cd /io/

/opt/python/cp310-cp310/bin/python3 -m pip install Cython==3.0.0a10 numpy scikit-learn && /opt/python/cp310-cp310/bin/python3 setup.py bdist_wheel
/opt/python/cp39-cp39/bin/python3 -m pip install Cython==3.0.0a10 numpy scikit-learn && /opt/python/cp39-cp39/bin/python3 setup.py bdist_wheel
/opt/python/cp38-cp38/bin/python3 -m pip install Cython==3.0.0a10 numpy scikit-learn && /opt/python/cp38-cp38/bin/python3 setup.py bdist_wheel
/opt/python/cp37-cp37m/bin/python3 -m pip install Cython==3.0.0a10 numpy scikit-learn && /opt/python/cp37-cp37m/bin/python3 setup.py bdist_wheel

for f in dist/*.whl; do auditwheel repair --plat manylinux2014_x86_64 $f; done
