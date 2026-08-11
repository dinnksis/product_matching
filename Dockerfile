FROM odsai/ecup26-matching-baseline:1.0

WORKDIR /solution

COPY run.py metadata.json ./
COPY src ./src
COPY model ./model

ENTRYPOINT ["python", "-u", "run.py"]

