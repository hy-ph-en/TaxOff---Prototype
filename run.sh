echo "$1"

export OPENAI_API_KEY=$1

pip install -r requirements.txt

streamlit run frontend/Home.py