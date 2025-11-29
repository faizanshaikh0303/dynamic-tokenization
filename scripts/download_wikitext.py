# scripts/download_wikitext.py
import os
import json
from datasets import load_dataset

OUTPUT_PATH = "data/wikitext_1gb.jsonl"
TARGET_BYTES = 1_000_000_000  # ~1GB
MIN_CHARS = 20

os.makedirs("data", exist_ok=True)

print("Streaming Wikitext-103...")
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)

written_bytes = 0
count = 0

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for item in dataset:
        text = item["text"].replace("\n", " ").strip()
        if len(text) < MIN_CHARS:
            continue
        line = json.dumps({"text": text}, ensure_ascii=False) + "\n"
        f.write(line)
        written_bytes += len(line.encode("utf-8"))
        count += 1
        if count % 50000 == 0:
            print(f"Processed {count} docs, {written_bytes/1e6:.2f} MB written")
        if written_bytes >= TARGET_BYTES:
            break

print(f"Done: {written_bytes/1e6:.2f} MB from {count} docs saved to {OUTPUT_PATH}")
