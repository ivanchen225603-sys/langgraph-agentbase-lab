FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.lock pyproject.toml ./
RUN python -m pip install --no-cache-dir -r requirements.lock
COPY agentbase ./agentbase
COPY raglab ./raglab
COPY scripts ./scripts
COPY data/knowledge ./data/knowledge
RUN python -m pip install --no-deps --no-build-isolation . && useradd --uid 10001 --create-home lab && mkdir /state && chown lab:lab /state
USER 10001:10001
EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health/ready', timeout=3)"
CMD ["sh", "-c", "python -m scripts.bootstrap_index && exec python -m raglab"]
