# Visions of Emotion

An AI pipeline that converts a spoken audio recording into generative artwork conditioned on the detected emotional content.  Originally created for a Pint of Science talk in 2022, discussing the impact of emotional content on artwork and the disparity between AI systems and human artists. 

**Pipeline:** WebM audio → FFmpeg WAV extraction → CNN emotion classifier → Conditional GAN → 10 generated images returned as base64 JSON.

---

## API

### `POST /detect_emotion`

**Request**
- Body: raw WebM audio binary
- `Content-Type: application/octet-stream`
- Max size: 10 MB
- Rate limit: 10 requests per 60 seconds per IP

**Response**
```json
{
  "vector": "[anger, disgust, fear, happiness, sadness]",
  "images": ["<base64>", ...]
}
```

| Status | Meaning |
|---|---|
| 200 | Success with images |
| 204 | Success, no images produced |
| 400 | Processing error |
| 411 | Missing `Content-Length` header |
| 413 | File too large |
| 429 | Rate limit exceeded |

---

## Models

| Model | File | Purpose |
|---|---|---|
| Audio CNN | `dataxxx.pt` | 4-layer CNN — classifies audio into a 5-value emotion vector (anger, disgust, fear, happiness, sadness) |
| Conditional GAN | `G-axial-emotion-one-hot-256-lr-1-full-dataset-e-300.pth` | Generates 512×512 images conditioned on the emotion vector |

Both weight files are baked into the container image at build time.

---

## Running locally

```bash
docker build -t visions-of-emotion .
docker run -p 8080:8080 visions-of-emotion
```

Test the endpoint:
```bash
curl -X POST http://localhost:8080/detect_emotion \
  --data-binary @recording.webm \
  -H "Content-Type: application/octet-stream"
```

**System requirements (outside Docker):** Python 3.11, `ffmpeg`, `sox`, `libsndfile1`.

