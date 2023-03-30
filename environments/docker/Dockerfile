
# ARG UBUNTU_VER=22.04
ARG CONDA_VER=latest
ARG OS_TYPE=x86_64
ARG PY_VER=3.9
ARG TF_VER=2.10.0

FROM  nvidia/cuda:12.0.1-cudnn8-devel-ubuntu22.04


# System packages 
RUN apt-get update && apt-get install -yq curl wget jq vim

# Use the above args during building https://docs.docker.com/engine/reference/builder/#understand-how-arg-and-from-interact
ARG CONDA_VER
ARG OS_TYPE
# Install miniconda to /miniconda
RUN curl -LO "http://repo.continuum.io/miniconda/Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh"
RUN bash Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh -p /miniconda -b
RUN rm Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda

ARG PY_VER
ARG TF_VER
# Install packages from conda and downgrade py (optional).
RUN conda install -c anaconda -y python=${PY_VER}
RUN conda install -c anaconda -y \
    pytorch torchvision torchaudio cudatoolkit=11.3.1 -c pytorch &&\
    conda install -c conda-forge wandb  &&\
    conda install -c anaconda jupyter &&\
    jupyter notebook --generate-config 

COPY ./jupyter_config/jupyter_notebook_config.py /root/.jupyter


    # conda-forge wandb \
    # pytorch torchvision torchaudio cudatoolkit=12.0.1 -c pytorch
    # && pip install pyyaml pandas

# Set the working directory and expose the Jupyter port
WORKDIR /workspace
EXPOSE 8888

# Start the Jupyter server
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]