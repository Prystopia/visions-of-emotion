# Visions of Emotion

An AI pipeline that converts a spoken audio recording into generative artwork conditioned on the detected emotional content.  Originally created for a Pint of Science talk in 2022, discussing the impact of emotional content on artwork and the disparity between AI systems and human artists. 

## Example output
![image one](/assets/0-0-live.png)
![image two](/assets/0-1-live.png)
![image three](/assets/3-1-live.png)
![image four](/assets/3-2-live.png)

**Pipeline:** WebM audio (fetched from S3) → FFmpeg WAV extraction → CNN emotion classifier → Conditional GAN → 10 generated images uploaded to S3 and returned as presigned URLs.

---

## API

### `POST /detect_emotion`

**Request**
- Body: JSON
- `Content-Type: application/json`
- Max size: 10 MB
- Rate limit: 10 requests per 60 seconds per IP

```json
{
  "audio_url": "<bucket-name>/<object-key>",
  "auth_key": "<your-auth-key>"
}
```

**Response**
```json
{
  "vector": "[anger, disgust, fear, happiness, sadness]",
  "images": ["<presigned-s3-url>", ...]
}
```

| Status | Meaning |
|---|---|
| 200 | Success with images |
| 204 | Success, no images produced |
| 400 | Missing or invalid request fields |
| 401 | Invalid auth key |
| 411 | Missing `Content-Length` header |
| 413 | File too large |
| 429 | Rate limit exceeded |
| 500 | Unexpected error |

---

## Models

| Model | File | Purpose |
|---|---|---|
| Audio CNN | `dataxxx.pt` | 4-layer CNN — classifies audio into a 5-value emotion vector (anger, disgust, fear, happiness, sadness) |
| Conditional GAN | `G-axial-emotion-one-hot-256-lr-1-full-dataset-e-300.pth` | Generates 512×512 images conditioned on the emotion vector |

Both weight files are baked into the container image at build time.

---

## Running locally

Copy `.env.example` to `.env` and fill in your values:

```
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_DEFAULT_REGION=
CORS_ORIGINS=http://localhost:3000
```

Then build and run:

```bash
docker build -t visions-of-emotion .
docker run -p 8080:8080 --env-file .env visions-of-emotion
```

Test the endpoint:
```bash
curl -X POST http://localhost:8080/detect_emotion \
  -H "Content-Type: application/json" \
  -d '{"audio_url": "my-bucket/my-audio-file", "auth_key": "my-auth-key"}'
```

**System requirements (outside Docker):** Python 3.11, `ffmpeg`, `sox`, `libsndfile1`.

