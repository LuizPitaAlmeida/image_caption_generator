# Image Caption Generator

This repository contains an executable paper that runs a simple experiment
in image captioning. To do this it make use of computer vision and natural
language process techniques such as CNN and Transformer models.

## Repository structure

**Folders**

- [executable_papers](./executable_papers): Notebooks with image captioning examples
- [src](./src): Python code with scripts for image captioning

**Files**

- [dockerfile](./dockerfile): Dockerfile to build computational environment
- [good_practices](./good_practices.md): Some good practices in reproducible research
- [license](./LICENSE): Apache 2.0 License
- [README.md](,/README.md): Main information and instruction to install and use scripts

## Setup Environment

These installation instructions were designed for Linux (Ubuntu) operational systems.
For other systems please refer to
[Docker Installation Documentation](https://docs.docker.com/get-docker/)

### Install Docker

For an easy installation, in a Linux terminal run:

```bash
$ sudo apt-get update
$ sudo apt-get install --yes docker.io
```

Test Docker installation:

```bash
$ sudo docker run --rm hello-world
```

Although it is easy, Docker Inc. recommends a different installation way.
For more details refer to
[Docker Installation Documentation](https://docs.docker.com/get-docker/)

## Build Environment

From the root directory of repository, run:

```bash
$ sudo docker build --tag img_caption --file Dockerfile --pull .
```

## Run Executable Papers

From the root directory of repository, run:

```bash
$ sudo docker run -it --rm -p 8888:8888 \
    --volume "$PWD"/executable_papers:/executable_papers \
    img_caption
```

After Control+Click in the prompt link, the notebook directory
will open in your browser. Chose one of the options and run all cells.
