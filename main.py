from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch, os, base64, json, logging, tempfile, sys
import moviepy.video.io.ffmpeg_tools as audioTools
from ConditionalGAN.run import run_art_generation
from SoundClassifier.run import run_audio_classifier
import uuid, re
from middleware.request_limit import MAX_UPLOAD_BYTES, LimitUploadSize
from middleware.rate_limit import RateLimitMiddleware
import os
import boto3
from boto3.s3.transfer import S3UploadFailedError
from botocore.exceptions import ClientError
import bcrypt


import io

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
app = FastAPI()

origins = os.environ.get("CORS_ORIGINS", "").split(",")
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["POST"])

app.add_middleware(LimitUploadSize)
#currently simple in memory approach, can build out if needed
app.add_middleware(RateLimitMiddleware)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/detect_emotion")
async def detect_emotion(request: Request):
    path = None
    try:
        req_body = await request.body()

        if len(req_body) > MAX_UPLOAD_BYTES:
            return Response(status_code=413)

        body = json.loads(req_body.decode("utf-8"))
        url = body.get("audio_url")
        auth_key = body.get("auth_key")

        vals = url.split("/")
        container_name = vals[0]
        identifier = vals[1]

        if not url or not auth_key:
            raise HTTPException(status_code=400, detail="audio_url and auth_key are required")

        if not re.fullmatch(r"[a-zA-Z0-9_\-]+", auth_key):
            raise HTTPException(status_code=400, detail="Invalid auth_key format")

        if not re.fullmatch(r"[a-zA-Z0-9_\-]+", identifier):
            raise HTTPException(status_code=400, detail="Invalid audio_url format")

        s3_resource = boto3.resource("s3")

        bucket = s3_resource.Bucket(container_name)

        auth_obj = bucket.Object(os.path.basename(f"{identifier}_auth_key.json"))
        auth_data = io.BytesIO()
        download_file(auth_obj, auth_data)

        auth_json = json.loads(auth_data.read().decode("utf-8"))

        if not bcrypt.checkpw(auth_key.encode("utf-8"), auth_json["auth_key"].encode("utf-8")):
            raise HTTPException(status_code=401, detail="Invalid auth key")

        audio_obj = bucket.Object(os.path.basename(f"{identifier}.webm"))
        data = io.BytesIO()
        download_file(audio_obj, data)

        req_id = str(uuid.uuid4())
        device = torch.device("cpu")

        path = os.path.join(tempfile.gettempdir(), f"audio-{req_id}.webm") 
        #save the file to disk
        with open(path, 'wb') as f:
            f.write(data.read())
        
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
            file_path = os.path.join(resultsFolderPath, filename)
            obj = bucket.Object(os.path.join(req_id,filename))
            try:
                obj.upload_file(file_path)
                print(f"Uploaded file {filename} into bucket {bucket.name} with key {obj.key}.")
                url = s3_resource.meta.client.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": bucket.name, "Key": obj.key},
                    ExpiresIn=3600  # URL valid for 1 hour
                )
                images.append(url)
            except S3UploadFailedError as err:
                print(f"Couldn't upload file {filename} to {bucket.name}.")
                print(f"\t{err}")
            os.remove(file_path)
        #remove empty directory
        os.rmdir(resultsFolderPath)
        result = {"vector": str(normalisedEmotions), "image_urls": images}
        status = 200 if images else 204
        return Response(content=json.dumps(result), media_type="application/json", status_code=status)
    except HTTPException:
        raise
    except Exception as ex:
        logging.exception(ex)
        return Response(content=str("Unexpected error occurred"), status_code=500)
    finally:
        if os.path.exists(path):
            os.remove(path)
        wav_path = path.replace("webm", "wav")
        if os.path.exists(wav_path):
            os.remove(wav_path)

    
def normalise(max, min, val):
    denominator = max - min
    return 0.0 if denominator == 0 else (val - min) / denominator

def download_file(obj, data):
    try:
        obj.download_fileobj(data)
        data.seek(0)
    except ClientError as err:
        print(f"Couldn't download {obj.key}.")
        print(
            f"\t{err.response['Error']['Code']}:{err.response['Error']['Message']}"
        )