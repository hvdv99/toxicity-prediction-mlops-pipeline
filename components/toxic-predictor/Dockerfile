FROM python:3.10.7-slim
WORKDIR /pipelines/component/src
ADD src/component.py ./
ADD src/app.py ./
ADD requirements.txt ./
ENV PROJECT_ID=assignment1-402316
ENV MODEL_REPO=models_de2023_group1
EXPOSE 5001
RUN pip install -r requirements.txt