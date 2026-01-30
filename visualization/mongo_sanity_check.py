import pymongo

uri = input("Paste Mongo URI: ").strip()
client = pymongo.MongoClient(uri)
config.read("connection.ini")
client = pymongo.MongoClient(config["DEFAULT"]["database"])
return client["camera-counts"]

db = client["camera-counts"]
coll = db["combined_stats"]  # IMPORTANT: bracket syntax because of '-'

print("Collections:", db.list_collection_names())
print("Count:", coll.count_documents({}))
doc = coll.find_one({}, {"_id": 0, "object_id": 1, "timestamp": 1, "location": 1, "detected_type": 1})
print("Sample doc:", doc)
