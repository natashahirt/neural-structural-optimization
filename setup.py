# Copyright 2019 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import setuptools

# Updated requirements for modern TensorFlow compatibility
INSTALL_REQUIRES = [
    'absl-py>=1.0.0',
    'apache-beam>=2.40.0',  # For distributed training pipeline
    'autograd>=1.8.0',
    'nlopt>=2.7.0',  # For MMA optimizer
    'numpy>=1.23.5,<2.0.0',  # Compatible with TensorFlow 2.15.0
    'matplotlib>=3.6.0',
    'pillow>=9.0.0',
    'scipy>=1.10.0',
    'scikit-image>=0.20.0',
    'seaborn>=0.12.0',
    'xarray>=2023.1.0',
    'pandas>=1.5.0',  # Required by xarray
]

# Optional dependencies for enhanced performance
EXTRAS_REQUIRE = {
    'fast': [
        'scikit-sparse>=0.4.0',  # For faster sparse matrix operations
    ],
    'dev': [
        'pytest>=7.0.0',
        'pytest-cov>=4.0.0',
        'black>=22.0.0',
        'flake8>=5.0.0',
    ],
}

# Handle Python version compatibility
if sys.version_info[:2] < (3, 7):
    INSTALL_REQUIRES.append('dataclasses')

setuptools.setup(
    name='neural-structural-optimization',
    version='0.1.0',  # Updated version to reflect modernization
    license='Apache 2.0',
    author='Google LLC',
    author_email='noreply@google.com',
    description='Neural reparameterization for structural optimization with modern TensorFlow support',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    url='https://github.com/google-research/neural-structural-optimization',
    packages=setuptools.find_packages(),
    python_requires='>=3.7',  # Updated minimum Python version
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    keywords='topology optimization, structural optimization, neural networks, tensorflow',
)
