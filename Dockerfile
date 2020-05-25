# Base stage
FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install --yes \
    jupyter \
    python3-pip &&\
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install torch==1.5.0+cpu \
    torchvision==0.6.0+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html


EXPOSE 8888
ENTRYPOINT ["jupyter", "notebook", "--ip=0.0.0.0", \
            "--no-browser", "--allow-root", \
            "--notebook-dir=/executable_papers"]
