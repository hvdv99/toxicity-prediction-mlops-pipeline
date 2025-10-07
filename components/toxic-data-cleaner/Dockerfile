FROM python:3.10.7-slim
WORKDIR /pipelines/component/src
ADD src/component.py ./
ADD src/app.py ./
ADD requirements.txt ./
EXPOSE 5000
RUN pip install -r requirements.txt