import uvicorn
from fastapi import FastAPI
from dotenv import load_dotenv
import os
load_dotenv()


# load environment variables
port = os.environ["PORT"]
# initialize FastAPI
app = FastAPI()
@app.get("/")
def index():
    return {"data": "Hello there my friend"}
if __name__ == "__main__":
    uvicorn.run("main:app", port=port, reload=False)