FROM python:3.12-slim

# Use a fixed path for build steps
ENV HATCHLING_DIR=/opt/hatchling

# Just creating the directory
RUN mkdir -p ${HATCHLING_DIR}


# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget bzip2 ca-certificates \
    git gosu passwd && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* 

# Install Miniforge
# Set environment variables for Miniforge for the user
ENV CONDA_DIR=/opt/miniforge3
ENV PATH=${CONDA_DIR}/bin:${PATH}
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Copy local Miniforge installer if present in build context (optional)
COPY docker/miniforge_download/Miniforge3-Linux-x86_64.sh /tmp/miniforge.sh

# Use local installer if present, else download
RUN if [ -f /tmp/miniforge.sh ]; then \
      echo "Using local Miniforge installer"; \
    else \
      echo "Downloading Miniforge installer"; \
      wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh; \
    fi

# Install
RUN /bin/bash /tmp/miniforge.sh -b -p ${CONDA_DIR}

# Clean up
RUN rm /tmp/miniforge.sh && \
    conda clean --tarballs --index-cache --packages --yes && \
    find ${CONDA_DIR} -follow -type f -name '*.a' -delete && \
    find ${CONDA_DIR} -follow -type f -name '*.pyc' -delete && \
    conda clean --force-pkgs-dirs --all --yes

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Copy the current directory contents into the container
COPY . ${HATCHLING_DIR}

# Set working directory
WORKDIR ${HATCHLING_DIR}

# Copy the entrypoint script
RUN chmod +x ./docker/entrypoint.sh

# Install the package
RUN pip install .

# Use the entrypoint script (runs as root initially to handle permissions)
ENTRYPOINT ["./docker/entrypoint.sh"]

# Default command
CMD ["hatchling"]