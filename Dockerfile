FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy==2.0.2
COPY server.py jev.py index.html ./
# Starting data: copied onto the persistent volume on first boot (see seed_data_dir in server.py).
COPY candidates.json labels.json notes.json keywords.json jev_apps.json wordvec.tsv ./
ENV HOST=0.0.0.0 \
    PORT=8080 \
    DATA_DIR=/data \
    AUTOPILOT_HOURS=12 \
    PYTHONUNBUFFERED=1
EXPOSE 8080
CMD ["python", "server.py"]
