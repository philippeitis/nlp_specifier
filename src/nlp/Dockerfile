FROM python:3.10-slim-bullseye

RUN apt-get clean && apt-get update && apt-get install -y -qq \
    gcc \
    g++

RUN pip install spacy
RUN python -m spacy download en_core_web_lg

COPY requirements.txt /
RUN pip install -r /requirements.txt

COPY ./*.py /app/

ENV SERVER_PORT = 5000
CMD python -m app launch -p $SERVER_PORT
