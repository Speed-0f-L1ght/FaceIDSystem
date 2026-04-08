#Сборка
FROM python:3.14-slim AS builder

ENV VENV_PATH=/opt/venv

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \ 
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv ${VENV_PATH}
ENV PATH="${VENV_PATH}/bin:${PATH}"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY src/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

#Запуск
FROM python:3.14-slim AS runner

ENV VENV_PATH=/opt/venv

WORKDIR /app

RUN apt-get update && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*


COPY --from=builder ${VENV_PATH} ${VENV_PATH}
ENV PATH="${VENV_PATH}/bin:${PATH}"

COPY . .

EXPOSE 80

CMD ["python" , "src/Main.py"]
