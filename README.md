# RoboTaxiEnv Web Experiment

This folder contains the server for the online RoboTaxiEnv experiment of my thesis. It builds upon the [implementation](https://github.com/Pearl-UTexas/RobotaxiEnv/tree/masterBefore) provided by [Pearl-UTexas](https://github.com/Pearl-UTexas).

## Dependencies
- [Docker](https://www.docker.com/)
- [MongoDB](https://www.mongodb.com/)

## Installation
To install the requirements, run the following command:
```
pip install -r requirements.txt
```
## Connecting MongoDB
This project uses python-dotenv to provide the MongoDB connection string stored in the `MONGODB_URI` environment variable. To automatically add the string to your environment, create a file called `.env` in the root of the project and add the string to the file.:
```
MONGODB_URI=<mongodb_uri>
```

## Running the Test-Server
There are multiple ways to run the server. The easiest way is to run through uvicorn.
```
uvicorn app:app --reload --host 0.0.0.0 --port 5000
```
The application is running on http://localhost:5000.

If you want to test the production server (gunicorn) and the delivery of the certificates, you first need to provide the certificates for your domain in the folder `certs`. Then you can run the server with the following command:
```
gunicorn -w 1 --keyfile=./cert/private.key --certfile=./cert/certificate.crt -k uvicorn.workers.UvicornWorker -b 0.0.0.0:5000 app:app
```
## Docker Container
To deploy the software on your server, you first have to build the docker container with the following command:
```
docker build -t robotaxi_javascript:latest .
```
To run the container on your machine as a server, you need to attach the local file system, where you want to store the data, to the container. You might need root priviledges to run on these reserved ports. Run the docker container with the following command:
```
docker run -it -v "$(pwd)"/data:/app/data -p 443:443/tcp robotaxi_javascript:latest
```