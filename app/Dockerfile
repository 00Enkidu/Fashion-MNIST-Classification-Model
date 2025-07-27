FROM python:3.12-slim
LABEL authors="Enkidu00"

WORKDIR /app

COPY app/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app/. ./

RUN mkdir -p ~/.streamlit
RUN cp config.toml ~/.streamlit/config.toml
RUN cp credentials.toml ~/.streamlit/credentials.toml

EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]
CMD ["main.py"]

RUN ls -R /app