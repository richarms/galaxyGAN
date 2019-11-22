FROM tensorflow/tensorflow:nightly-py3

LABEL maintainer="Richard Armstrong <richarms@ska.ac.za>"

RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir matplotlib Pillow && \
    pip3 install --no-cache-dir imageio

WORKDIR /train
ENTRYPOINT python /train/models/train_model.py
