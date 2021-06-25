from setuptools import setup, find_packages
from distutils.core import Extension

DISTNAME = 'pyoml'
VERSION = '0.1.0'
PACKAGES = find_packages()
EXTENSIONS = []
DESCRIPTION = 'pyoml: Pyomo Optimization extensions for Machine Learning models'
LONG_DESCRIPTION = open('README.md').read()
AUTHOR = 'Carl Laird'
MAINTAINER_EMAIL = 'carldlaird@users.noreply.github.com'
LICENSE = 'Revised BSD'
URL = 'no-url-yet'

setuptools_kwargs = {
    'zip_safe': False,
    'scripts': [],
    'include_package_data': True,
    'install_requires' : ['pyomo>=5.6', 'numpy', 'pytest', 'pandas']
}

setup(name=DISTNAME,
      version=VERSION,
      packages=PACKAGES,
      ext_modules=EXTENSIONS,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      author=AUTHOR,
      maintainer_email=MAINTAINER_EMAIL,
      license=LICENSE,
      url=URL,
      python_requires='>=3.6',
      **setuptools_kwargs)
