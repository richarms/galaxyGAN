# galaxyGAN

## Unique synthetic radio galaxies

![title](https://github.com/richarms/galaxyGAN/blob/master/images/title.png)

Simulations of the radio sky for next-generation instruments and algorithms require models that are both cosmologically accurate and unique. This project tackles the latter. It is a data-augmentation; we take as input images of known extended radio sources, capture the features of this latent space, and generate images of 'new' radio galaxies.

## ETL - Train - Serve

A template for deep learning data science projects

![title](https://github.com/richarms/galaxyGAN/blob/master/images/ETL_Train_Serve.png)

This repository is both a project on its own, as well as a template for deep learning projects. Major modes of a deep learning data science project have been incorporated, from ETL (extract, transform, load) to Train to Serve, as well as the important sandbox activity through all of these, mostly conducted in jupyter notebooks, which is called here Explore. Each of these steps is performed by a separate docker container.

To build the project, run `docker-compose build`

### ETL

In the ETL mode, the immutable input sample is a heterogenous sample of extended radio sources from three different radio surveys, stored as .fits image files. .fits images are centred on the nominal position of the source, and then converted to uniformly-sized arrays of normalised pixels. Domain-relevant image augmentation is performed (i.e. rotations, translations through the abscissa and ordinate, images are *not* arbitrarily cropped or scaled). The resulting numpy array is then converted to a tensorflow dataset, shuffled and randomised.

### Train

In the Train mode, the deep convolutional generative adversarial network (GAN) model is trained. To run training, do:

`docker run -it --rm -v $(pwd)/models:/train/models -v $(pwd)/data:/train/data ds-dl/train`

### Serve

In the Serve step, the generative model that was trained in the previous mode is 'served' in two ways: as a streamlit app for individual user interaction with the served model and also using the RESTful API of *Tensorflow Serving* for programmatic, performant and defined interaction with the model. [ToDo] Before serving, the model is selectively pruned for faster inference.

## Airflow

In order to automate and monitor the full ETL - Train - Serve process, the airflow task manager is used, operating on Docker images directly. to set up airflow for the project, go into the airflow directory and edit the DAG template to point to the docker containers build above. Then, run `docker-compose up` in the airflow directory to launch airflow

![title](https://github.com/richarms/galaxyGAN/blob/master/images/airflow.png)
