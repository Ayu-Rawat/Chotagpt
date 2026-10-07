# NeetCode GPT documentation

This repository is an educational, from-scratch implementation of GPT
building blocks with a unified training and generation workflow.

## Start here

- [Setup and usage](SETUP_AND_USAGE.md) - install dependencies and run the
  pipeline and individual APIs.
- [Architecture](ARCHITECTURE.md) - understand the data flow and module
  boundaries.
- [Development guide](DEVELOPMENT.md) - repository conventions, validation,
  and how to extend the project.

## Scope

The source contains focused data/model exercises plus the `run_gpt.py`
pipeline. Most exercise modules expose their functionality through a class
named `Solution`; the main GPT model is the `model.gpt.GPT` PyTorch module.
There is no checked-in dataset, web server, or deployment configuration.

The `.gptvenv` directory is a local virtual environment and is ignored by
Git. It is not required: create a fresh environment using the instructions
in [Setup and usage](SETUP_AND_USAGE.md).
