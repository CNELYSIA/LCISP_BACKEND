from fastapi import FastAPI
import ee
from utils.ee_downloader import eeDownloader
ee.Initialize()
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

@app.post("/eeDownload")
async def ee_download(config: dict):
   strPath = await eeDownloader(config)
   return {"file": strPath}
