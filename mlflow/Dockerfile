# Defining base image
FROM python:3.8

# Installing packages from PyPi
RUN pip install --upgrade pip && \
    pip install mlflow==1.14.1 && \
    pip install psycopg2-binary==2.8.5 && \
    pip install boto3==1.15.16

# Defining start up command
EXPOSE 5000
ENTRYPOINT ["mlflow", "server"]
CMD ["--host", "0.0.0.0", "--port", "5000", "--workers", "4"]