# galaxyGAN

A template for deep learning data science projects

![title]("images/title.png")

This repository is both a project on its own, as well as a template for deep learning projects. Major modes of a deep learning data science project have been incorporated, from ETL (extract, transform, load) to Train to Serve, as well as the important sandbox activity through all of these, mostly conducted in jupyter notebooks, which is called here Explore. Each of these steps is performed by a separate docker container.

In the ETL step, a heterogenous sample of extended radio sources from three different radio surveys, stored as .fits image files. .fits images are centred on the nominal position of the source, and then converted to uniformly-sized arrays of normalised pixels. Domain-relevant image augmentation is performed (i.e. rotations, translations through the abscissa and ordinate, images are not arbitrarily cropped or scaled). The resulting numpy array is then converted to a tensorflow dataset, shuffled and randomised.

In the Train mode, the deep convolutional generative adversarial network (GAN) model is trained.

to run training, do:

docker run -it --rm -v $(pwd)/models:/train/models -v $(pwd)/data:/train/data ds-dl/train

In the Serve step, the generative model is 'served' in two ways: as a streamlit app for individual user interaction with the served model and also using the RESTful API of tensorflow serving for programmatic, performant and defined interaction with the model. Optionally, before serving, the model is selectively pruned for faster inference.

In order to automate and monitor the full ETL - Train - Serve process, the airflow task manager is used, operating on Docker images directly.


