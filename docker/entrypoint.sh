#!/bin/bash
set -e

echo "Setting up environment..."

# Get configuration from environment variables with fallbacks
USER_ID=${USER_ID:-1000}
GROUP_ID=${GROUP_ID:-1000}
USER_NAME=${USER_NAME:-HatchlingUser}

echo "Using USER_ID=${USER_ID}, GROUP_ID=${GROUP_ID}, USER_NAME=${USER_NAME}"

# If we're running as root, create user and switch to it
if [ "$(id -u)" = "0" ]; then
    echo "Running as root, setting up user ${USER_NAME} with UID ${USER_ID} and GID ${GROUP_ID}"
    
    # Create group and user (simple approach)
    groupadd -g ${GROUP_ID} ${USER_NAME} 2>/dev/null || true
    useradd -m -u ${USER_ID} -g ${GROUP_ID} ${USER_NAME} 2>/dev/null || true

    # Allow HatchlingUser passwordless sudo for apt only (after user is created)
    echo "${USER_NAME} ALL=(ALL) NOPASSWD:/usr/bin/apt,/usr/bin/apt-get" >> /etc/sudoers

    # Creating hatchling directories
    mkdir -p /home/${USER_NAME}/.hatch
    mkdir -p /home/${USER_NAME}/.local

    # Creating user-specific conda/mamba directories
    mkdir -p /home/${USER_NAME}/.mamba/pkgs
    mkdir -p /home/${USER_NAME}/.conda
    
    # Create the environments.txt file that conda expects
    touch /home/${USER_NAME}/.conda/environments.txt

    # Fix ownership of mounted volumes
    chown -R ${USER_NAME}:${USER_NAME} /home/${USER_NAME}/.hatch 2>/dev/null || true
    chown -R ${USER_NAME}:${USER_NAME} /home/${USER_NAME}/.local 2>/dev/null || true
    chown -R ${USER_NAME}:${USER_NAME} /home/${USER_NAME}/.mamba 2>/dev/null || true
    chown -R ${USER_NAME}:${USER_NAME} /home/${USER_NAME}/.conda 2>/dev/null || true

    # Give the user ownership of the entire Miniforge installation
    echo "Fixing ownership of Miniforge installation..."
    chown -R ${USER_NAME}:${USER_NAME} /opt/miniforge3 2>/dev/null || true

    # Switch to user and execute command
    exec gosu ${USER_NAME} "$@"
else
    echo "Running as user $(whoami)"
    exec "$@"
fi