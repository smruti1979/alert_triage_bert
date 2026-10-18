FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN useradd -m -u 10001 app
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# For the transformer model: pip install --no-cache-dir -r requirements-train.txt
COPY triage ./triage
COPY artifacts ./artifacts
USER app
ENV OPERATING_POINT=/srv/artifacts/baseline/operating_point.json AUDIT_LOG=/tmp/audit.jsonl
EXPOSE 8080
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8080/healthz')"
CMD ["uvicorn", "triage.serve:app", "--host", "0.0.0.0", "--port", "8080"]
