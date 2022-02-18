# Defining base image
FROM python:3.8

# Installing packages from PyPi
RUN pip install mlflow==1.14.1 && \
    pip install boto3==1.15.16 && \
    pip install sklearn

WORKDIR /app

COPY train.py .

ENTRYPOINT ["python", "train.py"]