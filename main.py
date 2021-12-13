import uvicorn
from fastapi import FastAPI
import os
from dotenv import load_dotenv
load_dotenv()
# load environment variables
port = os.environ["PORT"]
# initialize FastAPI
app = FastAPI()
@app.get("/")
def index():
    return {"data": "Hello there"}
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)