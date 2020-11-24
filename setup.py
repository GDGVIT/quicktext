# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from io import open
from configparser import ConfigParser

# Always prefer setuptools over distutils
from setuptools import find_packages, setup

try:
    import builtins
except ImportError:
    import __builtin__ as builtins

# https://packaging.python.org/guides/single-sourcing-package-version/
# http://blog.ionelmc.ro/2014/05/25/python-packaging/
PATH_ROOT = os.path.dirname(__file__)

# import quicktext  # noqa: E402


def load_requirements(path_dir=PATH_ROOT, file_name='requirements.txt', comment_char='#'):
    with open(os.path.join(path_dir, file_name), 'r') as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        # filer all comments
        if comment_char in ln:
            ln = ln[:ln.index(comment_char)].strip()
        # skip directly installed dependencies
        if ln.startswith('http'):
            continue
        if ln:  # if requirement is not empty
            reqs.append(ln)


    return reqs


def load_long_description():
    # https://github.com/PyTorchLightning/pytorch-lightning/raw/master/docs/source/_images/lightning_module/pt_to_pl.png
    url = os.path.join(cfg['git_url'] ,'raw', cfg['version'] ,'docs')
    text = open('README.md', encoding='utf-8').read()
    # replace relative repository path to absolute link to the release
    text = text.replace('](docs', f']({url}')
    # SVG images are not readable on PyPI, so replace them  with PNG
    text = text.replace('.svg', '.png')
    return text

config = ConfigParser(delimiters=["="])
config.read("settings.ini")
cfg = config["DEFAULT"]

extras = {

}

setup(
    name='quicktext',
    version=cfg['version'],
    description=cfg['docs'],
    author=cfg['author'],
    author_email=cfg['author_email'],
    url=cfg['git_url'],
    download_url='https://github.com/GDGVIT/quicktext',
    license=cfg['license'],
    packages=find_packages(exclude=['tests', 'tests/*', 'benchmarks']),

    long_description=load_long_description(),
    long_description_content_type='text/markdown',
    include_package_data=True,
    zip_safe=False,

    keywords=['deep learning', 'pytorch', 'AI'],
    python_requires='>=3.6',
    setup_requires=[],
    install_requires=load_requirements(),
    extras_require=extras,
    project_urls={
        "Bug Tracker": "https://github.com/GDGVIT/quicktext/issues",
        "Documentation": "https://pytorch-lightning.rtfd.io/en/latest/",
        "Source Code": "https://github.com/GDGVIT/quicktext",
    },

    classifiers=[
        'Environment :: Console',
        'Natural Language :: English',
        # How mature is this project? Common values are
        #   3 - Alpha, 4 - Beta, 5 - Production/Stable
        'Development Status :: 4 - Beta',
        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Information Analysis',
        # Pick your license as you wish
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)