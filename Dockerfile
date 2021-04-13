# Defining base image
FROM python:3.9.4-slim

# Installing packages from PyPi
RUN pip install mlflow[extras]==1.15.0 && \
    pip install psycopg2-binary==2.8.6 && \
    pip install boto3==1.17.50

# Defining start up command
EXPOSE 5000
ENTRYPOINT ["mlflow", "server"]