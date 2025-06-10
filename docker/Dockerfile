FROM python:3.12-slim

# Install git and gosu (for safe user switching)
RUN apt-get update && apt-get install -y git gosu && rm -rf /var/lib/apt/lists/*

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
