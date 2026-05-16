import time
import os
import json
import numpy as np
from transformers import pipeline

def evaluate_sentiment():
    print("Loading sentiment engine...")
    model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    
    # Measure load time
    load_start = time.perf_counter()
    sentiment_model = pipeline("zero-shot-classification", model=model_name)
    load_end = time.perf_counter()
    load_time_ms = (load_end - load_start) * 1000
    
    SENTIMENT_DEFINITIONS = {
        "gratitude":      "grateful, happy and satisfied with the food",
        "disappointment": "disappointed with the food quality or quantity",
        "disgust":         "disgusted, food was rotten, moldy, or a health safety hazard",
        "frustration":     "frustrated with the merchant, pickup experience, or store being closed",
        "excitement":      "excited about an amazing deal or massive surprise find",
        "urgency":         "anxious or urgent about food expiring extremely soon"
    }
    LABEL_TO_ID = {v: k for k, v in SENTIMENT_DEFINITIONS.items()}
    CANDIDATE_LABELS = list(SENTIMENT_DEFINITIONS.values())
    
    test_data = [
        ("Thank you so much! The food is perfectly fresh and we are very happy.", "gratitude"),
        ("شكرا جزيلا! الطعام طازج تمامًا ونحن سعداء جدًا.", "gratitude"),
        ("The bread was completely moldy and disgusting, it's a huge health hazard.", "disgust"),
        ("كان الخبز متعفنًا ومقرفًا تمامًا، إنه خطر صحي كبير.", "disgust"),
        ("I showed up at the store but it was already closed, very frustrating experience.", "frustration"),
        ("ذهبت إلى المتجر ولكنه كان مغلقًا بالفعل، تجربة محبطة للغاية.", "frustration"),
        ("Wow, what an incredible find! We got so much food for such a great price.", "excitement"),
        ("يا للروعة، ياله من اكتشاف مذهل! حصلنا على الكثير من الطعام بسعر رائع.", "excitement"),
        ("This milk expires tomorrow, we need to pick it up urgently before it goes bad.", "urgency"),
        ("ينتهي تاريخ صلاحية هذا الحليب غدًا، نحتاج إلى استلامه بشكل عاجل قبل أن يفسد.", "urgency"),
        ("A bit disappointed with the portion sizes, expected a bit more.", "disappointment"),
        ("أشعر بخيبة أمل قليلاً من حجم الحصص، كنت أتوقع أكثر من ذلك بقليل.", "disappointment")
    ]

    
    # Cold start test
    cold_start_time = time.perf_counter()
    _ = sentiment_model("Cold start test", candidate_labels=CANDIDATE_LABELS, multi_label=True)
    cold_end_time = time.perf_counter()
    cold_ms = (cold_end_time - cold_start_time) * 1000

    print(f"Testing {len(test_data)} unique sentences (5 iterations each)...")
    
    latencies = []
    correct = 0
    total = 0
    
    for _ in range(5):
        for text, true_tag in test_data:
            start_time = time.perf_counter()
            result = sentiment_model(text, candidate_labels=CANDIDATE_LABELS, multi_label=True)
            end_time = time.perf_counter()
            
            latencies.append((end_time - start_time) * 1000)
            
            # Check accuracy (top prediction)
            top_label = result['labels'][0]
            pred_tag = LABEL_TO_ID[top_label]
            if pred_tag == true_tag:
                correct += 1
            total += 1
            
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    accuracy = (correct / total) * 100
    
    report = f"""
### Sentiment & Moderation Analysis (mDeBERTa)
| Metric | Value |
| :--- | :--- |
| **Model** | `{model_name}` |
| **Accuracy (Top-1)** | {accuracy:.1f}% |
| **Avg Latency** | {avg_latency:.2f} ms |
| **P95 Latency** | {p95_latency:.2f} ms |
| **Cold Start (1st Inf)** | {cold_ms:.2f} ms |
| **Model Load Time** | {load_time_ms:.2f} ms |
| **Samples Tested** | {total} |

#### Sample Evaluation
| Text | Expected Label |
| :--- | :--- |
| Thank you so much! The food is perfectly fresh and we are very happy. | gratitude |
| شكرا جزيلا! الطعام طازج تمامًا ونحن سعداء جدًا. | gratitude |
| The bread was completely moldy and disgusting, it's a huge health hazard. | disgust |
| كان الخبز متعفنًا ومقرفًا تمامًا، إنه خطر صحي كبير. | disgust |
"""
    
    output_file = os.path.join(os.path.dirname(__file__), '..', 'analysis.md')
    
    # Append to report
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write("\n" + report)
        
    print(f"Detailed analysis saved to {output_file}")

if __name__ == "__main__":
    evaluate_sentiment()
