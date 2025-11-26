"""Test recognize endpoint."""

import requests


def test_recognize():
    url = "http://localhost:8000/recognize"
    files = {'image': ('test_photo.jpg', open('test_photo.jpg', 'rb'), 'image/jpeg')}

    response = requests.post(url, files=files)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")

    # Check timing
    data = response.json()
    timing = data.get('processing_times', {})
    total = timing.get('total_ms', 0)
    print(f"Total time: {total:.1f}ms")

    assert total <= 4000, f"Too slow: {total}ms > 4000ms"
    assert 'products' in data
    assert 'totals' in data
    assert 'processing_times' in data

    print("Test passed!")


if __name__ == "__main__":
    test_recognize()
