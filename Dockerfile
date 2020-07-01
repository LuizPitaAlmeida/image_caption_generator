# Base stage
FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN pip install nltk==3.5 \
    pytorch-lightning==0.8.3 \
    transformers==3.0.0
RUN conda install -c conda-forge pycocotools

COPY src /src
RUN python /src/utils/setup.py install
RUN python /src/data_process/setup.py install
RUN python /src/models/setup.py install

EXPOSE 8888
