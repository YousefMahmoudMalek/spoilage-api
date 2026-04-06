import os
import sys

# Try to use the project's venv if available
venv_python = os.path.join(os.getcwd(), 'venv', 'Scripts', 'python.exe')

test_script = """
import logging
from transformers import pipeline

# Suppress warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

CANDIDATE_LABELS = [
    "grateful and satisfied with the food",
    "disappointed with the food quality",
    "disgusted, food was rotten or a health hazard",
    "frustrated with the merchant or pickup experience",
    "excited about a great deal or surprising find",
    "anxious or urgent about food expiring soon"
]

LABEL_MAP = {
    "grateful and satisfied with the food":              "gratitude",
    "disappointed with the food quality":                "disappointment",
    "disgusted, food was rotten or a health hazard":     "disgust",
    "frustrated with the merchant or pickup experience": "frustration",
    "excited about a great deal or surprising find":     "excitement",
    "anxious or urgent about food expiring soon":        "urgency"
}

def test_sentiment():
    print("Loading model 'facebook/bart-large-mnli' (~1.6GB)...")
    try:
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    test_cases = [
        "The food was great and I saved so much money!",
        "The bread was already moldy when I picked it up, very disappointed.",
        "Hurry up! The surprise bag expires in 10 minutes!",
        "I love that we are saving the planet one meal at a time.",
        "I arrived at the pickup time but the store was closed. Never again.",
        "Look at this amazing find! A full tray of fresh donuts!",
        "This is absolutely disgusting. The meat smells like rot and slime. Gross!",
        "The pickup was fine, thanks."
    ]

    print(f"{'Text':<50} | {'Mapped Labels (Score > 0.3)':<30}")
    print("-" * 90)
    
    for text in test_cases:
        result = classifier(text, candidate_labels=CANDIDATE_LABELS, multi_label=True)
        
        # Filter and map labels above 0.3 threshold
        qualified = []
        for label, score in zip(result["labels"], result["scores"]):
            if score > 0.3:
                mapped_id = LABEL_MAP.get(label, "unknown")
                qualified.append(f"{mapped_id}({score:.2f})")
        
        if not qualified:
            labels_str = "[neutral: true]"
        else:
            labels_str = ", ".join(qualified)
            
        print(f"{text[:47]:<50} | {labels_str:<30}")

if __name__ == '__main__':
    test_sentiment()
"""

with open('temp_sentiment_test.py', 'w', encoding='utf-8') as f:
    f.write(test_script)

print("Running BART sentiment test...")
import subprocess
try:
    # Run with a 10-minute timeout for the large model download if needed
    subprocess.run([venv_python, 'temp_sentiment_test.py'], check=True, timeout=600)
except Exception as e:
    print(f"Test failed or timed out. Output: {e}")
finally:
    if os.path.exists('temp_sentiment_test.py'):
        os.remove('temp_sentiment_test.py')
