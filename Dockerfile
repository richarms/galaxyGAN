
FROM jupyter/scipy-notebook

LABEL maintainer="Richard Armstrong <richarms@ska.ac.za>"

# Install Tensorflow
RUN conda install --quiet --yes \
    'tensorflow=1.15*' \
    'keras=2.3*' && \
    conda clean -tipsy && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER