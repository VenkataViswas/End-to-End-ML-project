FROM python:3.8-slim

WORKDIR /app

COPY . /app

RUN apt update -y && apt install awscli -y

RUN pip install -e .

COPY . .

CMD ["python3", "application.py"]