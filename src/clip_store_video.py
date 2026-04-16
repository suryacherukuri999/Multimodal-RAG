## surya - CLIP-based video frame vector store
## Stores CLIP embeddings of video keyframes with temporal metadata
import os
import open_clip
import torch
import faiss
import numpy as np
import pickle
from PIL import Image


class CLIPVideoStore:
    def __init__(self, persist_dir="faiss_store_videos"):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        self.metadata = []  # [{image_path, timestamp_sec, frame_index, video_name, video_path}]

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="laion2b_s32b_b82k"
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-L-14")
        self.model.eval()
        print("[INFO] CLIP model loaded for video frames: ViT-L-14")

    def _embed_image(self, image_path):
        """Embed a single image and return normalized numpy vector."""
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0)
        with torch.no_grad():
            emb = self.model.encode_image(image_tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.squeeze().numpy()

    def build_from_frames(self, frame_metadata_list, dedup_threshold=0.98):
        """Build FAISS index from extracted video keyframes with CLIP-based deduplication.

        Args:
            frame_metadata_list: list of dicts from video_extractor_video.save_keyframes()
                Each dict has: image_path, timestamp_sec, frame_index, video_name, video_path
            dedup_threshold: cosine similarity threshold above which frames are considered duplicates
        """
        print(f"[INFO] Building video CLIP index from {len(frame_metadata_list)} frames...")
        embeddings = []
        valid_meta = []

        for meta in frame_metadata_list:
            try:
                emb = self._embed_image(meta["image_path"])
                embeddings.append(emb)
                valid_meta.append(meta)
                print(f"[DEBUG] Embedded frame: {meta['image_path']}")
            except Exception as e:
                print(f"[ERROR] Failed to embed {meta['image_path']}: {e}")

        if not embeddings:
            print("[WARN] No frames were embedded.")
            return

        # CLIP-based deduplication: remove near-duplicate frames
        embeddings_np = np.array(embeddings).astype("float32")
        keep_indices = self._deduplicate(embeddings_np, dedup_threshold)

        embeddings_deduped = embeddings_np[keep_indices]
        meta_deduped = [valid_meta[i] for i in keep_indices]

        print(f"[INFO] After deduplication: {len(meta_deduped)}/{len(valid_meta)} frames kept "
              f"(threshold={dedup_threshold})")

        dim = embeddings_deduped.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings_deduped)
        self.metadata = [{
            "image_path": m["image_path"],
            "timestamp_sec": m["timestamp_sec"],
            "frame_index": m["frame_index"],
            "video_name": m["video_name"],
            "video_path": m["video_path"],
            "type": "video_frame"
        } for m in meta_deduped]

        self.save()
        print(f"[INFO] Video CLIP index built with {len(meta_deduped)} frames.")

    def _deduplicate(self, embeddings, threshold):
        """Remove near-duplicate frames using cosine similarity.
        Returns indices of frames to keep."""
        n = len(embeddings)
        if n == 0:
            return []

        keep = [0]  # always keep first frame
        for i in range(1, n):
            is_duplicate = False
            for j in keep:
                sim = np.dot(embeddings[i], embeddings[j])
                if sim > threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                keep.append(i)
        return keep

    def query_text(self, query, top_k=5):
        """Query the video frame index with text using CLIP."""
        if self.index is None or self.index.ntotal == 0:
            return []

        print(f"[INFO] Video CLIP query: '{query}'")

        prompts = [
            query,
            f"a video frame showing {query}",
            f"a scene with {query}",
            f"a photo of {query}",
        ]

        tokens = self.tokenizer(prompts)
        with torch.no_grad():
            text_emb = self.model.encode_text(tokens)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
            text_emb = text_emb.mean(dim=0, keepdim=True)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

        query_np = text_emb.cpu().numpy().astype("float32")
        actual_k = min(top_k, self.index.ntotal)
        D, I = self.index.search(query_np, actual_k)

        results = []
        for idx, score in zip(I[0], D[0]):
            if idx >= 0 and idx < len(self.metadata) and score > -1e30:
                results.append({"score": float(score), "metadata": self.metadata[idx]})
        return results

    def save(self):
        faiss.write_index(self.index, os.path.join(self.persist_dir, "clip_video.index"))
        with open(os.path.join(self.persist_dir, "clip_video_meta.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self):
        self.index = faiss.read_index(os.path.join(self.persist_dir, "clip_video.index"))
        with open(os.path.join(self.persist_dir, "clip_video_meta.pkl"), "rb") as f:
            self.metadata = pickle.load(f)
        print(f"[INFO] Video CLIP index loaded from {self.persist_dir}: {len(self.metadata)} frames")
## surya end
