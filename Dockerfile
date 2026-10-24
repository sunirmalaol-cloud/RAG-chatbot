FROM  python:3.11

WORKDIR /app_2

COPY app_2.py .

COPY .env .

COPY requirements.txt .

COPY Attention.pdf .

RUN python -m pip install --no-cache-dir --timeout 120 --retries 10 -r requirements.txt

EXPOSE 8501

CMD ["python","-m","streamlit","run","app_2.py"]