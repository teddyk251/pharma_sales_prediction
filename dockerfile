FROM python:3.9
EXPOSE 8501
COPY . /app
WORKDIR /app
RUN pip3 install -r ./requirements.txt
CMD ["streamlit", "run", "./app.py"]