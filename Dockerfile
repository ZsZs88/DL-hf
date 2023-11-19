FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update &&  \
	apt install -y  \
	openssh-server \
	git \
	libgl1 \
	nano

ENV HOME /home/custom_user
RUN mkdir -p $HOME/dl-hf

RUN mkdir -p $HOME/dl-hf/celeba
RUN mkdir -p $HOME/dl-hf/celeba/data
RUN mkdir -p $HOME/dl-hf/celeba/figs
RUN mkdir -p $HOME/dl-hf/celeba/samples
RUN mkdir -p $HOME/dl-hf/celeba/models
RUN mkdir -p $HOME/dl-hf/celeba/logs

RUN mkdir -p $HOME/dl-hf/danbooru
RUN mkdir -p $HOME/dl-hf/danbooru/data
RUN mkdir -p $HOME/dl-hf/danbooru/figs
RUN mkdir -p $HOME/dl-hf/danbooru/samples
RUN mkdir -p $HOME/dl-hf/danbooru/models
RUN mkdir -p $HOME/dl-hf/danbooru/logs

WORKDIR $HOME/dl-hf
ENV CELEBA=${CELEBA:-$HOME/dl-hf/celeba}
ENV DANBOORU=${DANBOORU:-$HOME/dl-hf/danbooru}

RUN pip freeze

COPY ./src ./src
COPY ./sshd_config /etc/ssh/sshd_config
COPY requirements.txt requirements.txt
COPY requirements-cuda.txt requirements-cuda.txt

EXPOSE 22
EXPOSE 8888

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-cuda.txt

RUN useradd -rm -d $HOME -s /bin/bash -g root -G sudo -u 1000 custom_user
RUN echo 'custom_user:unstable-diffusion' | chpasswd

SHELL ["/bin/bash", "-l", "-c"]

# Start Jupyter lab with custom password
ENTRYPOINT service ssh start && jupyter-lab \
	--ip 0.0.0.0 \
	--port 8888 \
	--no-browser \
	--NotebookApp.notebook_dir='$home/src' \
	--ServerApp.terminado_settings="shell_command=['/bin/bash']" \
	--allow-root