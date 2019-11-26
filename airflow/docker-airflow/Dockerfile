FROM puckel/docker-airflow:1.10.4

USER root
RUN groupadd --gid 999 docker \
    && usermod -aG docker airflow \
    && pip install docker \
    && pip install "apache-airflow[kubernetes]"
USER airflow
