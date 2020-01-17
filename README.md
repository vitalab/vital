# VITAL

Welcome to the git repo of the [Videos & Images Theory and Analytics Laboratory (VITAL)](http://vital.dinf.usherbrooke.ca/ "VITAL home page")
of Sherbrooke University, headed by Professor [Pierre-Marc Jodoin](http://info.usherbrooke.ca/pmjodoin/).

## How to use
This repository was not designed to be used as a standalone project, but was rather meant to be included inside other
repositories. The recommended way of including this repository is as a [git submodule](https://git-scm.com/book/en/v2/Git-Tools-Submodules)
inside the project from which you want to use it.

## Description
To help you follow along with the organization of the repository, here is a summary of each major package's purpose:

- `metrics`: common metrics that are not part of the traditional libraries, whether those metrics are losses for
training (see `metrics/train`) or scores to evaluate the systems' performance (see `metrics/evaluate`).

- `modules`: generic models, organized by task (e.g. `classification`, `generative`, etc.).

- `utils`: a wide range of common utilities that may be used by several models
(e.g. image processing, parameter groups, etc.).

## Requirements
The `vital.yml` file lists the dependencies required by the whole repository. In case you include the repository inside
your own project, you may want to add project specific dependencies, or maybe even remove some dependencies (if you only
use some of the utilities provided by the repository).
