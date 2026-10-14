import json
data = json.load(open('data/baseline_results.json'))
archetypes = data['clustering']['archetypes']
for a in archetypes:
    print(a["archetype_key"], ":", a["count"])
print("Total failures:", data['clustering']['num_failures'])