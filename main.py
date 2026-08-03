from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import torch, os, base64, json, logging, tempfile, sys
import moviepy.video.io.ffmpeg_tools as audioTools
from ConditionalGAN.run import run_art_generation
from SoundClassifier.run import run_audio_classifier
import uuid
from middleware.request_limit import MAX_UPLOAD_BYTES, LimitUploadSize
from middleware.rate_limit import RateLimitMiddleware
import os

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
app = FastAPI()

origins = os.environ.get("CORS_ORIGINS", "").split(",")
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["POST"])

app.add_middleware(LimitUploadSize)
#currently simple in memory approach, can build out if needed
app.add_middleware(RateLimitMiddleware)

@app.post("/detect_emotion")
async def detect_emotion(request: Request):
    path = os.path.join(tempfile.gettempdir(), "audio.webm") 
    try:
        req_body = await request.body()

        if len(req_body) > MAX_UPLOAD_BYTES:
            return Response(status_code=413)

        req_id = str(uuid.uuidv4())
        device = torch.device("cpu")
        #save the file to disk
        with open(path, 'wb') as f:
            f.write(req_body)
        
        audioTools.ffmpeg_extract_audio(path, path.replace("webm", "wav"))
        detectedEmotions = run_audio_classifier(path.replace("webm", "wav"), device)
        maxV = max(detectedEmotions)
        minV = min(detectedEmotions)
        normalisedEmotions = [normalise(maxV,minV, t) for t in detectedEmotions]

        normalisedEmotions[0]*=2
        normalisedEmotions[1]/=2
        normalisedEmotions[3]*=2
        normalisedEmotions[4]/=2

        #folder should be specific to the incoming request otherwise multiple requests would potentially delete other people's images
        resultsFolderPath = os.path.join(tempfile.gettempdir(), "images", req_id)
        os.makedirs(resultsFolderPath, exist_ok=True)
        run_art_generation(len(normalisedEmotions), device, normalisedEmotions, 5, 2, resultsFolderPath)

        #get all images in the folder
        images = []
        for filename in os.listdir(resultsFolderPath):
            with open(os.path.join(resultsFolderPath, filename), "rb") as w:
                data = w.read()
                images.append(base64.b64encode(data).decode("utf-8"))
            #delete the file
            os.remove(os.path.join(resultsFolderPath, filename))
        #remove empty directory
        os.rmdir(resultsFolderPath)
        result = {"vector": str(normalisedEmotions), "images": images}
        status = 200 if images else 204
        return Response(content=json.dumps(result), media_type="application/json", status_code=status)
    except Exception as ex:
        logging.exception(ex)
        return Response(content=str("Unexpected error occurred"), status_code=400)
    finally:
        #remove original audio file
        os.remove(path)
        #remove converted audio file
        os.remove(path.replace("webm", "wav"))

    
def normalise(max, min, val):
    denominator = max - min
    return 0.0 if denominator == 0 else (val - min) / denominator