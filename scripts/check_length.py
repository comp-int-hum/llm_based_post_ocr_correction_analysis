import json

with open("work/initial_comparison.json", "r") as in_file:
    files = json.load(in_file)

print(len(files))
