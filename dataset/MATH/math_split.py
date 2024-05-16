from pprint import pprint

import jsonlines as jsl

records = list(jsl.open("MATH-full.jsonl"))

# split records into 500-length chunks
chunks = [records[i : i + 100] for i in range(0, len(records), 200)]

# save with _pt1.jsonl suffix
for i, chunk in enumerate(chunks):
    with jsl.open(f"MATH-full_pt{i+1}.jsonl", "w") as f:
        f.write_all(chunk)
