FROM python:3

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY . /app
WORKDIR /app

COPY requirements.txt .
RUN pip --no-cache-dir install -r requirements.txt

COPY . /app
