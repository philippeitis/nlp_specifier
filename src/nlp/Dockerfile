FROM python:3.10-slim-bullseye

WORKDIR app

RUN pip install spacy
RUN python -m spacy download en_core_web_lg

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY ./*.py ./app/
COPY ./visualization ./app/visualization

ENV CORE_SERVICE_PORT = 5000
CMD python -m app launch -p $CORE_SERVICE_PORT
