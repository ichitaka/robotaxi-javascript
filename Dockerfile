# syntax=docker/dockerfile:1
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

# set env variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY . .

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -r requirements.txt

CMD ["gunicorn", "-w", "1", "--keyfile=./cert/private.key", "--certfile=./cert/certificate.crt", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:443", "app:app"]
EXPOSE 443