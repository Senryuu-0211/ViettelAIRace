import json
from collections import Counter

data = [json.loads(line) for line in open('d:/ViettelAIRace/Project_3/trace-round1.jsonl')]

# Multi-turn analysis
print('=== MULTI-TURN CONVERSATION ANALYSIS (first 25 requests) ===')
for i, d in enumerate(data[:25]):
    msgs = d['body']['messages']
    roles = [m['role'] for m in msgs]
    lens = [len(m['content'])//4 for m in msgs]
    role_str = ' -> '.join(f'{r}({l}tok)' for r, l in zip(roles, lens))
    print(f'Req {i} (t={d["timestamp_ms"]}ms): {len(msgs)} msgs: {role_str}')

# Conversation length vs arrival time
print('\n=== CONVERSATION LENGTH vs ARRIVAL TIME ===')
for i in [0, 19, 20, 39, 40, 59, 60, 79, 80, 99, 100, 119]:
    d = data[i]
    msgs = d['body']['messages']
    total_tokens = sum(len(m['content']) for m in msgs) // 4
    print(f'Req {i} (t={d["timestamp_ms"]}ms): {len(msgs)} msgs, ~{total_tokens} tokens')

# User message prefix sharing
print('\n=== USER MESSAGE PREFIX SHARING ===')
user_prefixes_500 = []
for d in data:
    msgs = d['body']['messages']
    user_msgs = [m['content'] for m in msgs if m['role'] == 'user']
    combined = user_msgs[0][:500] if user_msgs else ''
    user_prefixes_500.append(combined)

prefix_counts = Counter(user_prefixes_500)
print(f'Unique user first-msg prefixes (500 chars): {len(prefix_counts)}')
for prefix, count in prefix_counts.most_common(5):
    print(f'  Count={count}: first 80 chars = "{prefix[:80]}"')

# Batch content similarity
print('\n=== BATCH CONTENT SIMILARITY ===')
all_first_user = [d['body']['messages'][1]['content'][:200] for d in data]
unique_all = len(set(all_first_user))
print(f'All 120 requests: {unique_all} unique user first messages (200 chars)')

# Check message structure patterns
print('\n=== MESSAGE STRUCTURE PATTERNS ===')
msg_patterns = Counter()
for d in data:
    msgs = d['body']['messages']
    pattern = tuple(m['role'] for m in msgs)
    msg_patterns[pattern] += 1

for pattern, count in msg_patterns.most_common(10):
    print(f'  Count={count}: {" -> ".join(pattern)}')

# Token distribution histogram
print('\n=== TOKEN DISTRIBUTION ===')
all_tokens = []
for d in data:
    msgs = d['body']['messages']
    total = sum(len(m['content']) for m in msgs) // 4
    all_tokens.append(total)

buckets = [(0, 20000), (20000, 25000), (25000, 30000), (30000, 35000), (35000, 40000), (40000, 50000)]
for lo, hi in buckets:
    count = sum(1 for t in all_tokens if lo <= t < hi)
    print(f'  {lo//1000}K-{hi//1000}K tokens: {count} requests')

# Check if conversations are continuations (same first user message = same conversation)
print('\n=== CONVERSATION CONTINUATION ANALYSIS ===')
first_user_msgs = {}
for i, d in enumerate(data):
    msgs = d['body']['messages']
    first_user = msgs[1]['content'][:100]
    if first_user not in first_user_msgs:
        first_user_msgs[first_user] = []
    first_user_msgs[first_user].append((i, len(msgs), d['timestamp_ms']))

for prefix, reqs in list(first_user_msgs.items())[:5]:
    if len(reqs) > 1:
        print(f'  Shared prefix "{prefix[:60]}...":')
        for req_id, num_msgs, ts in reqs:
            print(f'    Req {req_id}: {num_msgs} msgs, t={ts}ms')
