# Food Photo Analyzer

AI-powered food photo analysis API that detects ingredients and calculates nutrition data.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set the OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key"
```

3. Run the server:
```bash
uvicorn main:app --reload
# or for production:
uvicorn main:app --host 0.0.0.0 --port 8080
```

## Docker Deployment

```bash
# Build image
docker build -t food-photo-analyzer .

# Run container
docker run --env OPENAI_API_KEY=your_key_here -p 8080:8080 food-photo-analyzer
```

## API Usage

### Analyze Photo

**POST** `/analyze`

Analyze a photo to extract ingredients and nutrition information.

#### Request
- **Method**: POST
- **Content-Type**: `multipart/form-data`
- **Body**: Form field `image` containing the photo file
- **Acceptable formats**: JPEG, PNG

#### Example with curl
```bash
curl -X POST -F "image=@food_photo.jpg" http://localhost:8080/analyze
```

#### Example with Python
```python
import requests

with open('food_photo.jpg', 'rb') as file:
    files = {'image': file}
    response = requests.post('http://localhost:8080/analyze', files=files)

print(response.json())
```

#### Response
```json
{
  "products": [
    {
      "product_name": "burger bun",
      "quantity_g": 80,
      "confidence": 0.95
    },
    {
      "product_name": "lettuce",
      "quantity_g": 15,
      "confidence": 0.9
    }
  ],
  "totals": {
    "kcal": 485,
    "protein": 34,
    "fat": 28,
    "carbs": 30
  }
}
```

### Health Check

**GET** `/health`

```bash
curl http://localhost:8080/health
# Returns: {"status": "ok"}
```

## Error Responses

All errors return status 422 with details:

```json
{"detail": "Image field is required"}
{"detail": "Unsupported format (use jpeg/png)"}
{"detail": "Analysis error: {error details}"}
