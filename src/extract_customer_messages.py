import csv
import json
import os

INPUT_PATH = r"D:\HiverAssignment\data\processed\amazon_cases.jsonl"
OUTPUT_PATH = r"D:\HiverAssignment\data\processed\customer_messages.csv"

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

with open(INPUT_PATH, "r", encoding="utf-8") as infile, \
     open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as outfile:

    writer = csv.DictWriter(
        outfile,
        fieldnames=[
            "message_id",
            "conversation_id",
            "timestamp",
            "text"
        ]
    )

    writer.writeheader()

    count = 0

    for line in infile:
        case = json.loads(line)

        for message in case["messages"]:
            if message["author"] == "customer":
                writer.writerow({
                    "message_id": message["tweet_id"],
                    "conversation_id": case["conversation_id"],
                    "timestamp": message["created_at"],
                    "text": message["text"]
                })

                count += 1

print(f"Customer messages extracted: {count:,}")
print(f"Output: {OUTPUT_PATH}")