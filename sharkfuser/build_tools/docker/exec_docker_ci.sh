#!/usr/bin/env bash

# ROCm requires accesses to the host's /dev/kfd and /dev/dri/* device nodes, typically
# owned by the `render` and `video` groups. The groups' GIDs in the container must
# match the host's to access the resources. Sometimes the device nodes may be owned by
# dynamic GIDs (that don't belong to the `render` or `video` groups). So instead of
# adding user to the GIDs of named groups (obtained from `getent group render` or
# `getent group video`), we simply check the owning GID of the device nodes on the host
# and pass it to `docker run` with `--group-add=<GID>`.
for DEV in /dev/kfd /dev/dri/*; do
  # Skip if not a character device
  # /dev/dri/by-path/ symlinks are ignored
  [[ -c "${DEV}" ]] || continue
  DOCKER_RUN_DEVICE_OPTS+=" --device=${DEV} --group-add=$(stat -c '%g' ${DEV})"
done

# Bind mounts for the following:
# - current directory to /workspace in the container
docker run --rm \
           ${DOCKER_RUN_DEVICE_OPTS} \
           -v "${PWD}":/workspace \
           ghcr.io/sjain-stanford/compiler-dev-ubuntu-24.04:main@sha256:5d70805daf1c9e89c3515e888fa54282982ce868784b13bb9c98f09d4fbd1c5b \
           "$@"
