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
    

    # Creating hatchling directories
    mkdir -p /home/${USER_NAME}/.hatch
    mkdir -p /home/${USER_NAME}/.local
    # Fix ownership of mounted volumes
    chown -R ${USER_NAME}:${USER_NAME} /home/${USER_NAME} 2>/dev/null || true
    
    chown -R ${USER_NAME}:${USER_NAME} /home/${USER_NAME}/.hatch 2>/dev/null || true
    chown -R ${USER_NAME}:${USER_NAME} /home/${USER_NAME}/.local 2>/dev/null || true
    # Switch to user and execute command
    exec gosu ${USER_NAME} "${@:-hatchling}"
else
    echo "Running as user $(whoami)"
    exec "${@:-hatchling}"
fi
