FROM ubuntu:20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV SUMO_HOME="/usr/share/sumo"
ENV LIBSUMO_AS_TRACI=1

# Install dependencies
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:sumo/stable && \
    apt-get update && \
    apt-get install -y \
        python3 \
        python3-pip \
        python3-dev \
        git \
        wget \
        sumo \
        sumo-tools \
        sumo-doc && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Install sumo-rl dependencies
COPY . /workspace/sumo-rl
RUN pip3 install -e /workspace/sumo-rl

# Install Jupyter Notebook
RUN pip3 install jupyter

# Expose port for Jupyter
EXPOSE 8888

# Set working directory
WORKDIR /workspace

# Start Bash by default
CMD ["bash"]