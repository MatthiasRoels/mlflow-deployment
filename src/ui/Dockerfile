# Defining base image
FROM python:3.8

# Installing packages from PyPi
RUN pip install mlflow==1.15.0

# Defining start up command
EXPOSE 5000
ENTRYPOINT ["mlflow", "ui"]
CMD ["--host", "0.0.0.0"]