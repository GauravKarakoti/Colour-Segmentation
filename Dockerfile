FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 libxkbcommon-x11-0 libqt5gui5 libqt5widgets5 && \
    rm -rf /var/lib/apt/lists/*

ENV DISPLAY=host.docker.internal:0.0

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python"]
CMD ["palette.py"]
