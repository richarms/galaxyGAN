import logging

from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
#from airflow.operators.docker_operator import DockerOperator
#from airflow.contrib.operators.kubernetes_pod_operator import KubernetesPodOperator

from datetime import datetime

import docker
import docker.client

log = logging.getLogger(__name__)

default_args = {
    'owner': 'richarms',
    'start_date': datetime(2019, 8, 29),
}

def read_xcoms(**context):
    for idx, task_id in enumerate(context['data_to_read']):
        data = context['task_instance'].xcom_pull(task_ids=task_id, key='data')
        logging.info(f'[{idx}] I have received data: {data} from task {task_id}')

def launch_docker_container(**context):
    image_name = context['image_name']
    client: Client = docker.from_env()

    log.info(f"Creating image {image_name}")
    container = client.create_container(image=image_name)

    container_id = container.get('Id')
    log.info(f"Running container with id {container_id}")
    client.start(container=container_id)

    logs = client.logs(container_id, follow=True, stderr=True, stdout=True, stream=True, tail='all')

    try:
        while True:
            l = next(logs)
            log.info(f"Task log: {l}")
    except StopIteration:
        pass
        
    inspect = client.inspect_container(container)
    log.info(inspect)
    if inspect['State']['ExitCode'] != 0:
                raise Exception("Container has not finished with exit code 0")

    log.info(f"Task ends!")
    my_id = context['my_id']
    context['task_instance'].xcom_push('data', f'my name is {my_id}', context['execution_date'])


def do_test_docker():
    client = docker.from_env()
    for container in client.containers():
        logging.info(str(container))

with DAG('pipeline_docker', default_args=default_args) as dag:
    t1 = BashOperator(
        task_id='print_date1',
        bash_command='date')

    t2_1_id = 'do_task_one'
    t2_1 = PythonOperator(
        task_id=t2_1_id,
        provide_context=True,
        op_kwargs={
            'image_name': 'task1',
            'my_id': t2_1_id
        },
        python_callable=launch_docker_container
    )

    t2_2_id = 'do_task_four'
    t2_2 = PythonOperator(
        task_id=t2_2_id,
        provide_context=True,
        op_kwargs={
            'image_name': 'task4',
            'my_id': t2_2_id
        },
        python_callable=launch_docker_container
    )
    
#    t2_3_id = 'do_docker_container'
#    t2_3 = DockerOperator(
#        task_id = t2_3_id,
#        image="task4:latest",
#        api_version='auto',
#        auto_remove=True,
#        command="/bin/sleep 30",
#        docker_url="unix://var/run/docker.sock",
#        network_mode="bridge"
#    )

#    t2_4_id = "do_kubernetes"
#    t2_4 = KubernetesPodOperator(
#        namespace='default',
#        image="python:3.7",
#        cmds=["python","-c"],
#        arguments=["print('hello world')"],
#        labels={"foo": "bar"},
#        name="kube_pass",
#        task_id=t2_4_id,
#        get_logs=True,
#    )

    t3 = PythonOperator(
        task_id='read_xcoms',
        provide_context=True,
        python_callable=read_xcoms,
        op_kwargs={
            'data_to_read': [t2_1_id, t2_2_id]
        }
    )

    t1_5 = PythonOperator(
        task_id="test_docker",
        python_callable=do_test_docker
    )

    t1 >> t1_5 >> [t2_1, t2_2] >> t3
