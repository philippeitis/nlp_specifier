FROM python:3.10-slim-bullseye

COPY requirements.txt /
RUN pip install -r /requirements.txt

COPY ./*.py /app/

ENV UI_SERVICE_PORT=5001
ENV STREAMLIT_SERVER_PORT=$UI_SERVICE_PORT

ENTRYPOINT ["streamlit", "run"]
CMD ["/app/__init__.py"]