FROM python:3.7

COPY ./requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

COPY ./app /app

RUN useradd -m myuser
USER myuser

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8080"]