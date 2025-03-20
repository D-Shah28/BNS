import json

with open("university_map.geojson", "r", encoding="utf-8") as file:
    university_map = json.load(file)

locations = [feature["properties"].get("name", "Unnamed") for feature in university_map["features"]]

print("Available Destinations in GeoJSON:")
for loc in locations:
    print(f"- {loc}")
