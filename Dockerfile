FROM python:3.10.13

WORKDIR ./

COPY . .

RUN pip install -r requirements.txt

EXPOSE 8501

ENV OPENAI_API_KEY=${OPENAI_API_KEY}

ENTRYPOINT ["streamlit", "run", "frontend/Home.py", "--server.port=8501", "--server.address=0.0.0.0"]

# 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