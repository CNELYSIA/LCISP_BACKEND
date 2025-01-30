from fastapi import FastAPI
import ee
from starlette.middleware.cors import CORSMiddleware

from utils.ee_downloader import eeDownloader
ee.Initialize()
app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:5173",
    "http://localhost:3000",
]

# noinspection PyTypeChecker
app.add_middleware(

    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
