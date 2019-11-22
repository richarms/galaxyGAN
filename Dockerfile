FROM tensorflow/tensorflow:nightly-py3

LABEL maintainer="Richard Armstrong <richarms@ska.ac.za>"

WORKDIR /train
ENTRYPOINT python /train/models/train_model.py
