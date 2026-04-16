import chromadb
client = chromadb.PersistentClient(path="/home/ubuntu/Double_Check_AI/storage/vector_db")
collection = client.get_collection(name="procurements")

# Kustutame need kaks faili, mis jäid poolikuks
files_to_fix = ["112072025025.akt", "113042023003.akt"]
for f in files_to_fix:
    collection.delete(where={"source": f})
    print(f"Puhastatud: {f}")