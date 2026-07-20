import json

data = [json.loads(line) for line in open('d:/ViettelAIRace/Project_3/trace-round1.jsonl')]

# Timestamp analysis
timestamps = [d['timestamp_ms'] for d in data]
print('=== ARRIVAL PATTERN ===')
print(f'First arrival: {timestamps[0]}ms')
print(f'Last arrival: {timestamps[-1]}ms')
print(f'Total duration: {timestamps[-1] - timestamps[0]}ms = {(timestamps[-1] - timestamps[0])/1000:.1f}s')

# Inter-arrival times
gaps = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
print(f'Min gap: {min(gaps)}ms')
print(f'Max gap: {max(gaps)}ms')
print(f'Avg gap: {sum(gaps)/len(gaps):.1f}ms')

# Burst detection
from collections import Counter
windows = [t // 1000 for t in timestamps]
window_counts = Counter(windows)
print(f'Max requests in 1s window: {max(window_counts.values())}')
print(f'Number of 1s windows with requests: {len(window_counts)}')
print(f'First 20 timestamps: {timestamps[:20]}')
print(f'Last 10 timestamps: {timestamps[-10:]}')

# Arrival distribution per 5-second window
windows_5s = [t // 5000 for t in timestamps]
w5_counts = Counter(windows_5s)
print('\n=== ARRIVAL PER 5-SECOND WINDOW ===')
for w in sorted(w5_counts.keys()):
    print(f'  {w*5}s - {(w+1)*5}s: {w5_counts[w]} requests')

# Token estimation
print('\n=== PROMPT SIZE ANALYSIS ===')
sys_content = data[0]['body']['messages'][0]['content']
sys_tokens_est = len(sys_content) // 4
print(f'System prompt: ~{sys_tokens_est} tokens ({len(sys_content)} chars)')

all_total = []
all_user = []
all_num_msgs = []
for d in data:
    msgs = d['body']['messages']
    user_len = sum(len(m['content']) for m in msgs if m['role'] == 'user')
    total_len = sum(len(m['content']) for m in msgs)
    all_total.append(total_len // 4)
    all_user.append(user_len // 4)
    all_num_msgs.append(len(msgs))

print(f'Total tokens - min:{min(all_total)}, max:{max(all_total)}, avg:{sum(all_total)//len(all_total)}')
print(f'User tokens  - min:{min(all_user)}, max:{max(all_user)}, avg:{sum(all_user)//len(all_user)}')
print(f'Num messages per req - min:{min(all_num_msgs)}, max:{max(all_num_msgs)}')

# Generation params
print('\n=== GENERATION PARAMS (first 5) ===')
for i, d in enumerate(data[:5]):
    body = d['body']
    max_tokens = body.get('max_tokens', body.get('max_completion_tokens', 'NOT SET'))
    temperature = body.get('temperature', 'NOT SET')
    stream = body.get('stream', 'NOT SET')
    top_p = body.get('top_p', 'NOT SET')
    print(f'Req {i}: max_tokens={max_tokens}, temp={temperature}, stream={stream}, top_p={top_p}')

# Check all unique generation params
all_max_tokens = set()
all_temps = set()
all_streams = set()
for d in data:
    body = d['body']
    all_max_tokens.add(body.get('max_tokens', body.get('max_completion_tokens', 'NOT SET')))
    all_temps.add(body.get('temperature', 'NOT SET'))
    all_streams.add(body.get('stream', 'NOT SET'))

print(f'\nUnique max_tokens: {all_max_tokens}')
print(f'Unique temperatures: {all_temps}')
print(f'Unique stream values: {all_streams}')

# Workload types
wl_types = Counter(d.get('workload_type', 'unknown') for d in data)
print(f'\nWorkload types: {dict(wl_types)}')

# Check if there are shared prefixes beyond system prompt
print('\n=== PREFIX SHARING ANALYSIS ===')
user_first_msgs = []
for d in data:
    msgs = d['body']['messages']
    for m in msgs:
        if m['role'] == 'user':
            user_first_msgs.append(m['content'][:200])
            break

# Check how many share the same first 200 chars of user message
prefix_counts = Counter(user_first_msgs)
print(f'Unique user message prefixes (first 200 chars): {len(prefix_counts)}')
if len(prefix_counts) < 20:
    for prefix, count in prefix_counts.most_common(10):
        print(f'  Count={count}: "{prefix[:80]}..."')
