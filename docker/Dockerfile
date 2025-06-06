FROM python:3.12-slim

# Install git
RUN apt-get update && apt-get install -y git

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

ARG USER_ID
ARG GROUP_ID
# Copy the current directory contents into the container at /app
COPY . /app

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
RUN groupadd -g ${GROUP_ID} appuser
RUN useradd -m -u ${USER_ID} -g ${GROUP_ID} appuser
RUN chown -R appuser:appuser /app
USER appuser

# Create working directory
WORKDIR /app

# Install the package
RUN pip install -e /app

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["/home/appuser/.local/bin/hatchling"]
