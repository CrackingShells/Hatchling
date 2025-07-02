FROM python:3.12-slim

# Set environment variables for Miniforge
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install Miniforge and required dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget bzip2 ca-certificates \
        git gosu passwd && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh -O /tmp/miniforge.sh && \
    /bin/bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniforge.sh && \
    conda clean --tarballs --index-cache --packages --yes && \
    find ${CONDA_DIR} -follow -type f -name '*.a' -delete && \
    find ${CONDA_DIR} -follow -type f -name '*.pyc' -delete && \
    conda clean --force-pkgs-dirs --all --yes

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Copy the current directory contents into the container at /app
COPY . /app

# Set working directory
WORKDIR /app

# Copy the entrypoint script
RUN chmod +x ./docker/entrypoint.sh

# Install the package
RUN pip install -e .

# Use the entrypoint script (runs as root initially to handle permissions)
ENTRYPOINT ["./docker/entrypoint.sh"]

# Default command
CMD ["hatchling"]