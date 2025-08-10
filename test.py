import torch
import json
from tqdm import tqdm
from FlagEmbedding.research.visual_bge.visual_bge.modeling import Visualized_BGE
from src.multimodal_retriever.retriever_v2 import RetrieverV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(pretrained_model_path="src/multimodal_retriever/pretrained_model/Visualized_m3.pth"):
    retriever = RetrieverV2(pretrained_model_path=pretrained_model_path)
    return retriever

def load_database(database_path:str="data/cleaned_vlsp2025_law_db.json"):
    with open(database_path,"r") as f:
        database = json.load(f)
    return database

def embedding_database(database, model):
    for record in tqdm(database, desc="Generating embeddings for the database"):
        # pt_embedding = model.encode_document(record["text"], convert_to_tensor=True, device=DEVICE)
        pt_embedding = model.encode_text(record["text"]).squeeze(0)
        np_embedding = pt_embedding.detach().cpu().numpy()
        list_embedding = np_embedding.tolist()
        record["embedding"] = list_embedding

    return database

def save_database(database_path:str, database):
    with open(database_path, "w") as f:
        json.dump(database, f, indent=4, ensure_ascii=False)
    print(f"Text embeddings and text saved to {database}")

if __name__=="__main__":

    # Load the model
    print("Loading the model...")
    model = load_model()
    print("Loaded model")
    print("-"*10)

    # Load the database
    print("Loading the database.....")
    database = load_database()
    print("Loaded the database...")

    # Embed the text of the database
    print("Embedding the database")
    database = embedding_database(database, model)
    print("Embedded the database")

    # Save teh database
    print("Saving the database...")
    save_database("data/record_id_to_document_embedding_v2.json", database)
    print("Saved the database")

