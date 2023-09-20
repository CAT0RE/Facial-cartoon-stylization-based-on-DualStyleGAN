import io
import os
import uuid

from PIL import Image
from fastapi import FastAPI, UploadFile
from loguru import logger
from sse_starlette.sse import ServerSentEvent, EventSourceResponse
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles

from util import save_image

app = FastAPI(
    title="CS3310 Project",
    description="人脸卡通化",
    version="0.1.0",
    middleware=[Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])]
)

if not os.path.exists("static"):
    os.mkdir("static")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
async def startup_event():
    from packed_model import Model, prepare_models
    prepare_models()
    app.package = {
        "model": Model(),
    }


@app.post("/api/upload")
async def upload(file: UploadFile):
    file_id = uuid.uuid4()
    # read file as PIL image
    image = Image.open(io.BytesIO(await file.read()))
    image.load()
    image.convert("RGB")
    # Save image to static
    image.save(f"static/{file_id}.png")
    return {"id": file_id}


@app.get("/api/transfer")
async def transfer(file_id: str, style_id: int, segment: bool, structure_only: bool):
    def generator():
        from packed_model import TransferEvent
        model = app.package["model"]
        for ev in model.transfer(f"static/{file_id}.png", style_id, segment, structure_only):
            if isinstance(ev, TransferEvent):
                logger.info("Transfer event: {}".format(ev.type))
                path = f"static/{file_id}_{ev.type}.png"
                save_image(ev.data, path)
                yield ServerSentEvent(
                    data=path,
                    event=ev.type,
                )

    return EventSourceResponse(generator())


@app.get("/")
async def index():
    return FileResponse("dist/index.html")


app.mount("/", StaticFiles(directory="dist"), name="dist")
