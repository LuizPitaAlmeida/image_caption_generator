# Image Caption Generator

This repository contains an executable paper that runs a simple experiment
in image captioning. To do this it make use of computer vision and natural
language process techniques such as CNN and Transformer models.

More info about algorithm refer to [Image_Captioning.pdf](./Image_Captioning.pdf).

For a reproducible paper of our results refer to
[Colab Notebook](<https://colab.research.google.com/drive/1oHJTiFP97rvTqZ6ye5kPHSRS7AmBd4mW?usp=sharing>)

Now, the reproduction only by Colab with GPU.
The Docker feature described here are under construction.

## Repository structure

**Folders**

- [data](./data): Links for data download.
- [executable_papers](./executable_papers): Notebooks with image captioning examples
- [figures](./figures): Figures used in the paper
- [src](./src): Python code with scripts for image captioning

**Files**

- [dockerfile](./dockerfile): Dockerfile to build computational environment
- [good_practices](./good_practices.md): Some good practices in reproducible research
- [Image_Captioning](./Image_Captioning.pdf): Research Paper
- [license](./LICENSE): Apache 2.0 License
- [README.md](,/README.md): Main information and instruction to install and use scripts

## Setup Environment

These installation instructions were designed for Linux (Ubuntu) operational systems.
For other systems please refer to
[Docker Installation Documentation](https://docs.docker.com/get-docker/)

### Install Docker [Under Construction]

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

## Build Environment [Under Construction]

From the root directory of repository, run:

```bash
$ sudo docker build --tag img_caption --file Dockerfile --pull .
```

## Run Executable Papers [Under Construction]

From the root directory of repository, run:

```bash
$ sudo docker run -it --rm -p 8888:8888 \
    --volume "$PWD"/executable_papers:/executable_papers \
    img_caption
```

After Control+Click in the prompt link, the notebook directory
will open in your browser. Chose one of the options and run all cells.
