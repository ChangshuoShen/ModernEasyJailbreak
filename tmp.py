data_path = "advbench_test.json"
import json

with open(data_path, 'r') as f:
    data = json.load(f)

for item in data:
    item['prompt'] = item.pop('query')
    item['target'] = item.pop('reference_responses')
    item['target'] = item['target'][0]

with open(data_path, 'w') as f:
    json.dump(data, f, indent=4)