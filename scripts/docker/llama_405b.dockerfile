FROM rocm/7.0-preview:rocm7.0_preview_pytorch_training_mi35X_alpha

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get install -y bash busybox coreutils
