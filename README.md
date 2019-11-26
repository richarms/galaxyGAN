# galaxyGAN

A template for deep learning data science projects

This repository is both a project on its own, as well as a template for deep learning projects. I have incorporated the major modes in a deep learning data science project, from ETL to Train to Serve, as well as the important sandbox activity through all of these, mostly conducted in jupyter notebooks, which we'll call Explore.

In the ETL step, we'll take in this case as raw data a heterogenous sample of extended radio sources from three different radio surveys, stored as .fits image files. We convert these images to uniformly-sized arrays, normalise, and centre the images. Domain-relevant image augmentation is performed: rotations, translations through the abscissa and ordinate (i.e. we do not arbitrarily crop or scale). The resulting shuffled and randomised numpy array is then converted to a tensorflow dataset.

In the train step, we train the model.

In the serve step, we serve the generated model in two ways: as a streamlit app for individual user interaction with the served model and also using the RESTful API of tensorflow serving for programmatic, performant and defined interaction with the model. Optionally, before serving, the model is selectively pruned for faster inference.

Each of these steps is performed by a separate docker container

In order to automate and monitor the full ETL - Train - Serve process, we use the airflow task manager operating on Docker images directly.
