# hash:sha256:ce8ec86c97d8dc88f67cd40c6723cb57d6bf5bd677dbfbcf85c1a8f20d54f46e
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN conda install -y \
        matplotlib==3.8.4 \
        numpy==1.23.5 \
        pandas==1.5.3 \
        scikit-learn==1.5.1 \
        scipy==1.10.1 \
        seaborn==0.13.2 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    allensdk==2.16.2 \
    brain-observatory-utilities==0.4.0 \
    multiprocess==0.70.16 \
    nbconvert==7.16.4
