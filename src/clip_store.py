## surya - CLIP-based image vector store
import os
import faiss
import numpy as np
import pickle
from PIL import Image
import open_clip
import torch

class CLIPImageStore:
    def __init__(self, persist_dir: str = "faiss_store_images"):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        self.metadata = []

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="laion2b_s32b_b82k"
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-L-14")
        self.model.eval()
        print("[INFO] CLIP model loaded: ViT-L-14")

    def build_from_images(self, image_paths: list):
        print(f"[INFO] Building CLIP index from {len(image_paths)} images...")
        embeddings = []
        valid_paths = []

        for path in image_paths:
            try:
                image = Image.open(path).convert("RGB")
                image_tensor = self.preprocess(image).unsqueeze(0)
                with torch.no_grad():
                    emb = self.model.encode_image(image_tensor)
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                embeddings.append(emb.squeeze().numpy())
                valid_paths.append(path)
                print(f"[DEBUG] Embedded: {path}")
            except Exception as e:
                print(f"[ERROR] Failed to embed {path}: {e}")

        if not embeddings:
            print("[WARN] No images were embedded.")
            return

        embeddings = np.array(embeddings).astype("float32")
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine similarity)
        self.index.add(embeddings)
        self.metadata = [{"image_path": p, "type": "image"} for p in valid_paths]
        self.save()
        print(f"[INFO] CLIP index built with {len(valid_paths)} images.")

    def query_text(self, query: str, top_k: int = 3):
        print(f"[INFO] CLIP query: '{query}'")

        prompts = [
            query,
            f"a photo of {query}",
            f"a photograph of {query}",
            f"an image showing {query}",
            f"a picture of {query}",
        ]

        tokens = self.tokenizer(prompts)
        with torch.no_grad():
            text_emb = self.model.encode_text(tokens)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

            # Average all prompt embeddings into a single query embedding
            text_emb = text_emb.mean(dim=0, keepdim=True)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

        query_np = text_emb.cpu().numpy().astype("float32")
        D, I = self.index.search(query_np, top_k)

        results = []
        for idx, score in zip(I[0], D[0]):
            if idx < len(self.metadata):
                results.append({"score": float(score), "metadata": self.metadata[idx]})
        return results


    def save(self):
        faiss.write_index(self.index, os.path.join(self.persist_dir, "clip.index"))
        with open(os.path.join(self.persist_dir, "clip_meta.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self):
        self.index = faiss.read_index(os.path.join(self.persist_dir, "clip.index"))
        with open(os.path.join(self.persist_dir, "clip_meta.pkl"), "rb") as f:
            self.metadata = pickle.load(f)
        print(f"[INFO] CLIP index loaded from {self.persist_dir}")
## surya end