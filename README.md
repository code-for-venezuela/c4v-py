# C4V-Py

## Installation

Use pip to install the package:
```
pip install c4v-py
```

## Development

The following tools are used in this project:
- [Poetry](https://python-poetry.org/) is used as package manager.
- [Nox](https://nox.thea.codes/) is used as automation tool, mainly for testing.
- [Black](https://black.readthedocs.io/) is the mandatory formatter tool.
- [PyEnv](https://github.com/pyenv/pyenv/wiki) is recommended as a tool to handle multiple python versions in your machine.

The library is intended to be compatible with python ~3.6.9, ~3.7.4 and ~3.8.2. But the primary version to support is ~3.8.2.

The general structure of the project is trying to follow the recommendations
in [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/).
The main difference lies in the source code itself which is not constraint to data science code.

## Pendings

- [ ] Change the authors field in pyproject.toml
- [ ] Change the repository field in pyproject.toml 
- [ ] Move the content below to a place near to the data in the data folder or use the reference folder.
Check [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) for details.
- [ ] Understand what is in the following folders and decide what to do with them.
    - [ ] brat-v1.3_Crunchy_Frog
    - [ ] creating_models
    - [X] data/data_to_annotate
    - [ ] data_analysis
- [ ] Set symbolic links between `brat-v1.3_Crunchy_Frog/data` and `data/data_to_annotate`.  `data_sampler` extracts to `data/data_to_annotate`.  Files placed here are read by Brat.
