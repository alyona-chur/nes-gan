FROM python:3.8-slim

WORKDIR /mlflow
RUN pip install --no-cache-dir --upgrade --progress-bar off pip
RUN pip install --no-cache-dir --progress-bar off mlflow==1.23.1 protobuf==3.20.0 numpy==1.19.5

EXPOSE 5000

ENV MLFLOW_ARTIFACT_URI /mlflow/artifacts
RUN mkdir -p ${MLFLOW_ARTIFACT_URI}
CMD ["mlflow", "server", \
     "--backend-store-uri", "/mlflow/mlruns", \
     "--default-artifact-root", "${MLFLOW_ARTIFACT_URI}", \
     "--host", "0.0.0.0"]
