import json
with open("a.json", "r") as jsonFile:
    data = json.load(jsonFile)

data["status"] = "Projecting"

with open("a.json", "w") as jsonFile:
    json.dump(data, jsonFile)