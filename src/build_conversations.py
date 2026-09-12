import json
import os

import pandas as pd


DATA_PATH = r"D:\HiverAssignment\data\raw\twcs\twcs.csv"
OUTPUT_PATH = r"D:\HiverAssignment\data\processed\amazon_cases.jsonl"

BRAND = "AmazonHelp"
CHUNK_SIZE = 100_000


print("Building support cases...")
print(f"Brand: {BRAND}")


# ---------------------------------------------------------
# 1. Read the dataset and build a lightweight tweet index
# ---------------------------------------------------------

tweets = {}

print("\nReading dataset in chunks...")

for chunk in pd.read_csv(
    DATA_PATH,
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "in_response_to_tweet_id",
    ],
    chunksize=CHUNK_SIZE,
):

    for row in chunk.itertuples(index=False):

        tweet_id = int(row.tweet_id)

        parent_id = None

        if pd.notna(row.in_response_to_tweet_id):
            parent_id = int(row.in_response_to_tweet_id)

        tweets[tweet_id] = {
            "tweet_id": tweet_id,
            "author_id": row.author_id,
            "inbound": bool(row.inbound),
            "created_at": row.created_at,
            "text": row.text,
            "parent_id": parent_id,
        }


print(f"Indexed {len(tweets):,} tweets.")


# ---------------------------------------------------------
# 2. Identify Amazon tweets
# ---------------------------------------------------------

amazon_tweets = {
    tweet_id
    for tweet_id, tweet in tweets.items()
    if tweet["author_id"] == BRAND
}

print(f"Amazon tweets: {len(amazon_tweets):,}")


# ---------------------------------------------------------
# 3. Build direct parent → children relationships
# ---------------------------------------------------------

children = {}

for tweet_id, tweet in tweets.items():

    parent_id = tweet["parent_id"]

    if parent_id is None:
        continue

    if parent_id not in children:
        children[parent_id] = []

    children[parent_id].append(tweet_id)


# ---------------------------------------------------------
# 4. Find customer tweets that Amazon directly answered
# ---------------------------------------------------------

customer_roots = set()

for amazon_id in amazon_tweets:

    parent_id = tweets[amazon_id]["parent_id"]

    if parent_id is None:
        continue

    if parent_id not in tweets:
        continue

    parent = tweets[parent_id]

    # The parent must be a customer tweet.
    if parent["inbound"]:
        customer_roots.add(parent_id)


print(
    f"Customer messages with direct Amazon replies: "
    f"{len(customer_roots):,}"
)


# ---------------------------------------------------------
# 5. Follow each direct reply chain
# ---------------------------------------------------------

def build_chain(start_id):

    chain = []

    current_id = start_id
    visited = set()

    while current_id is not None:

        if current_id in visited:
            break

        if current_id not in tweets:
            break

        visited.add(current_id)

        tweet = tweets[current_id]

        chain.append(
            {
                "tweet_id": tweet["tweet_id"],
                "author": (
                    "brand"
                    if tweet["author_id"] == BRAND
                    else "customer"
                ),
                "inbound": tweet["inbound"],
                "created_at": tweet["created_at"],
                "text": tweet["text"],
            }
        )

        # Find the next direct reply.
        next_replies = children.get(current_id, [])

        if not next_replies:
            break

        # Follow only one direct reply chain.
        #
        # Prefer a reply from the opposite side.
        next_id = None

        for candidate_id in sorted(
            next_replies,
            key=lambda x: tweets[x]["created_at"],
        ):

            candidate = tweets[candidate_id]

            if candidate["author_id"] == BRAND:
                if tweet["author_id"] != BRAND:
                    next_id = candidate_id
                    break

            else:
                if tweet["author_id"] == BRAND:
                    next_id = candidate_id
                    break

        if next_id is None:
            break

        current_id = next_id

    return chain


# ---------------------------------------------------------
# 6. Write cases as JSONL
# ---------------------------------------------------------

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True,
)

case_count = 0
message_count = 0

print("\nWriting support cases...")

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8",
) as output_file:

    for customer_id in customer_roots:

        chain = build_chain(customer_id)

        # We require at least:
        # customer → brand
        if len(chain) < 2:
            continue

        # Make sure the second message is actually Amazon.
        if chain[1]["author"] != "brand":
            continue

        case = {
            "conversation_id": customer_id,
            "brand": BRAND,
            "messages": chain,
        }

        output_file.write(
            json.dumps(
                case,
                ensure_ascii=False,
            )
            + "\n"
        )

        case_count += 1
        message_count += len(chain)


print("\n" + "=" * 70)
print("SUPPORT CASE BUILD COMPLETE")
print("=" * 70)

print(f"Cases:    {case_count:,}")
print(f"Messages: {message_count:,}")
print(f"Output:   {OUTPUT_PATH}")