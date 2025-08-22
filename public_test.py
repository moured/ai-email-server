"""Simple public test for candidate to run"""

import json
import requests

SERVER_URL = "http://localhost:8090"


def test_chat_completions():
    """Test OpenAI proxy endpoint"""
    url = f"{SERVER_URL}/v1/chat/completions"
    payload = {
        "model": "gpt-5-mini", # "deepseek/deepseek-r1-0528:free", "gpt-5-mini",
        "messages": [{"role": "user", "content": "Hello, respond with just 'Hi!'"}],
        "max_tokens": 10,
    }

    response = requests.post(url, json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    content = data["choices"][0]["message"]["content"].strip()
    print(f"✓ Chat completions: {content}")


def test_prefill_simple():
    """Test prefill endpoint with simple email"""
    url = f"{SERVER_URL}/v1/prefill"

    with open("emails/simple_invoice.txt", "r") as f:
        email_text = f.read()

    payload = {"email_text": email_text}
    response = requests.post(url, json=payload)

    assert response.status_code == 200
    data = response.json()
    print(f"✓ Prefill: {json.dumps(data, separators=(',', ':'))}")
    assert data["success"] is True

    # Show CSV
    import os
    import csv

    if os.path.exists("data.csv"):
        with open("data.csv", "r") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                print(f"Row {i + 1}: {json.dumps(dict(row), separators=(',', ':'))}")


def cleanup_csv():
    """Clean up CSV file after tests"""
    import os

    if os.path.exists("data.csv"):
        os.remove("data.csv")
        print("Cleaned up data.csv file")


if __name__ == "__main__":
    try:
        test_chat_completions()
        test_prefill_simple()
        print("All tests passed!")
    finally:
        cleanup_csv()
