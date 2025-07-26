#!/usr/bin/env bash

# Bind mounts for the following:
# - current directory to /src in the container
# - user's HOME directory (useful for .bash*, .gitconfig, .venv, .cache etc)
docker run -it \
           -v "${PWD}":"/src" \
           -v "${HOME}":"${HOME}" \
           sjainstanford/ubuntu-24.04-dev:latest
