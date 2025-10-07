FROM python:3.10.7-slim
WORKDIR /pipelines/component/src
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY src/app.py ./
COPY src/templates ./templates/
EXPOSE 5002