#!/bin/bash
set -e

echo "Setting up environment..."

# Get configuration from environment variables with fallbacks
USER_ID=${USER_ID:-1000}
GROUP_ID=${GROUP_ID:-1000}
USER_NAME=${USER_NAME:-appuser}

echo "Using USER_ID=${USER_ID}, GROUP_ID=${GROUP_ID}, USER_NAME=${USER_NAME}"

# If we're running as root, create user and switch to it
if [ "$(id -u)" = "0" ]; then
    echo "Running as root, setting up user..."
    
    # Create group and user (simple approach)
    groupadd -g ${GROUP_ID} ${USER_NAME} 2>/dev/null || true
    useradd -m -u ${USER_ID} -g ${GROUP_ID} ${USER_NAME} 2>/dev/null || true
    
    # Fix ownership of mounted volumes
    chown -R ${USER_NAME}:${USER_NAME} /home/${USER_NAME} 2>/dev/null || true

    # Ensure all of /app is owned by the target user
    chown -R ${USER_NAME}:${USER_NAME} /app 2>/dev/null || true
    
    # Switch to user and execute command
    exec gosu ${USER_NAME} "${@:-hatchling}"
else
    echo "Running as user $(whoami)"
    exec "${@:-hatchling}"
fi
