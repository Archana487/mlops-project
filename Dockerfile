FROM python:3.12-slim
WORKDIR /app
RUN useradd -m app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONPATH=/app/src PRP_ARTIFACTS_DIR=/data/artifacts PRP_REPORTS_DIR=/data/reports
RUN mkdir -p /data && chown -R app /data /app
USER app
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "prp.serve:app", "--host", "0.0.0.0", "--port", "8000"]
