import os, requests

images = {
    "flood": [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/2017_Barpeta_flood.jpg/320px-2017_Barpeta_flood.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d5/Flooding_in_Argentina.jpg/320px-Flooding_in_Argentina.jpg",
    ],
    "cyclone": [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/Hurricane_Isabel_from_ISS.jpg/320px-Hurricane_Isabel_from_ISS.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/54/Cyclone_Catarina_from_the_ISS_on_March_26_2004.JPG/320px-Cyclone_Catarina_from_the_ISS_on_March_26_2004.JPG",
    ],
    "disaster": [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/0/02/2010_Haiti_earthquake_damage.jpg/320px-2010_Haiti_earthquake_damage.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Devastation_in_Tacloban%2C_Leyte.jpg/320px-Devastation_in_Tacloban%2C_Leyte.jpg",
    ],
    "normal": [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/24701-nature-natural-beauty.jpg/320px-24701-nature-natural-beauty.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/Sunset_at_the_Salton_Sea.jpg/320px-Sunset_at_the_Salton_Sea.jpg",
    ],
}

for category, urls in images.items():
    for i, url in enumerate(urls):
        path = f"dataset/{category}/img_{i+1}.jpg"
        try:
            r = requests.get(url, timeout=10)
            with open(path, "wb") as f:
                f.write(r.content)
            print(f"✅ Downloaded: {path}")
        except Exception as e:
            print(f"❌ Failed: {url} → {e}")

print("\n✅ Dataset download complete!")