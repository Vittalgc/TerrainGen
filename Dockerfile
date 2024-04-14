FROM tensorflow/tensorflow:2.13.0-gpu

WORKDIR /app

RUN pip install -U -r requirements.txt

EXPOSE 8501

ENV NVIDIA_VISIBLE_DEVICES all

ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]