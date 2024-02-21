# simple fastapi server for testing, returns hello world
# run with: uvicorn test-server:app --reload
# test with: curl http://localhost:8000

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/hello/{name}")
def hello_name(name: str):
    return {"Hello": name}


@app.get("/hello")
def hello():
    return {"Hello": "World"}
