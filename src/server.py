"""
src/server.py — Flask API for federated RAG search.
Each node runs this server. All paths (data, indexes) come from the config file.

Run from project root:
    python -m src.server --config peers_a.json
    python -m src.server --config peers_b.json

Endpoints:
    POST /search/images/scores  — Phase 1: return scores + filenames (lightweight)
    POST /search/images/fetch   — Phase 2: return full image for a specific file
    POST /search/pdfs           — Single phase: return scores + text chunks
    GET  /health                — Health check
"""

import os
import json
import base64
import argparse
from io import BytesIO
from flask import Flask, request, jsonify
from PIL import Image

app = Flask(__name__)

# ── Globals (initialized in main) ──────────────────────────────────────────
clip_store = None
pdf_store = None
NODE_ID = "unknown"
THUMBNAIL_MAX_SIZE = (512, 512)


def load_config(config_path="peers.json"):
    with open(config_path, "r") as f:
        return json.load(f)


def make_thumbnail_b64(image_path, max_size=None):
    if max_size is None:
        max_size = THUMBNAIL_MAX_SIZE
    img = Image.open(image_path).convert("RGB")
    img.thumbnail(max_size, Image.LANCZOS)
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=80)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ── Image Endpoints ────────────────────────────────────────────────────────

@app.route("/search/images/scores", methods=["POST"])
def search_images_scores():
    if clip_store is None or clip_store.index is None:
        return jsonify({"error": "Image index not loaded", "node_id": NODE_ID}), 503

    data = request.get_json()
    query = data.get("query", "")
    top_k = data.get("top_k", 3)

    if not query:
        return jsonify({"error": "Missing 'query'"}), 400

    try:
        results = clip_store.query_text(query, top_k=top_k)
        response = []
        for r in results:
            path = r["metadata"]["image_path"]
            response.append({
                "score": round(r["score"], 6),
                "filename": os.path.basename(path),
                "node_id": NODE_ID
            })
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e), "node_id": NODE_ID}), 500


@app.route("/search/images/fetch", methods=["POST"])
def fetch_image():
    if clip_store is None:
        return jsonify({"error": "Image index not loaded", "node_id": NODE_ID}), 503

    data = request.get_json()
    filename = data.get("filename", "")

    if not filename:
        return jsonify({"error": "Missing 'filename'"}), 400

    target_path = None
    for meta in clip_store.metadata:
        if os.path.basename(meta["image_path"]) == filename:
            target_path = meta["image_path"]
            break

    if target_path is None or not os.path.exists(target_path):
        return jsonify({"error": f"Image '{filename}' not found", "node_id": NODE_ID}), 404

    try:
        b64 = make_thumbnail_b64(target_path)
        return jsonify({
            "image_base64": b64,
            "filename": filename,
            "node_id": NODE_ID
        })
    except Exception as e:
        return jsonify({"error": str(e), "node_id": NODE_ID}), 500


# ── PDF Endpoint ───────────────────────────────────────────────────────────

@app.route("/search/pdfs", methods=["POST"])
def search_pdfs():
    if pdf_store is None or pdf_store.index is None:
        return jsonify({"error": "PDF index not loaded", "node_id": NODE_ID}), 503

    data = request.get_json()
    query = data.get("query", "")
    top_k = data.get("top_k", 5)

    if not query:
        return jsonify({"error": "Missing 'query'"}), 400

    try:
        results = pdf_store.query(query, top_k=top_k)
        response = []
        for r in results:
            meta = r.get("metadata", {})
            response.append({
                "score": round(float(r["distance"]), 6),
                "text": meta.get("text", ""),
                "source": meta.get("source", "unknown"),
                "node_id": NODE_ID
            })
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e), "node_id": NODE_ID}), 500


# ── Health Check ───────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "node_id": NODE_ID,
        "status": "ok",
        "has_images": clip_store is not None and clip_store.index is not None,
        "has_pdfs": pdf_store is not None and pdf_store.index is not None,
        "image_count": len(clip_store.metadata) if clip_store and clip_store.metadata else 0,
    })


# ── Init ───────────────────────────────────────────────────────────────────

def init_stores(config):
    """Initialize stores using directories from config."""
    global clip_store, pdf_store

    data_dir = config.get("data_dir", "data")
    faiss_images_dir = config.get("faiss_store_images_dir", "faiss_store_images")
    faiss_pdfs_dir = config.get("faiss_store_dir", "faiss_store")

    # ── CLIP image index ──
    clip_index_path = os.path.join(faiss_images_dir, "clip.index")
    clip_meta_path = os.path.join(faiss_images_dir, "clip_meta.pkl")
    if os.path.exists(clip_index_path) and os.path.exists(clip_meta_path):
        from src.clip_store import CLIPImageStore
        clip_store = CLIPImageStore(persist_dir=faiss_images_dir)
        clip_store.load()
        print(f"[SERVER] CLIP index loaded from '{faiss_images_dir}': {len(clip_store.metadata)} images")
    else:
        from src.data_loader import load_images
        image_paths = load_images(data_dir)
        if image_paths:
            from src.clip_store import CLIPImageStore
            clip_store = CLIPImageStore(persist_dir=faiss_images_dir)
            clip_store.build_from_images(image_paths)
            print(f"[SERVER] CLIP index built from '{data_dir}': {len(clip_store.metadata)} images")
        else:
            print(f"[SERVER] No images in '{data_dir}' — image search disabled")

    # ── PDF index ──
    faiss_path = os.path.join(faiss_pdfs_dir, "faiss.index")
    meta_path = os.path.join(faiss_pdfs_dir, "metadata.pkl")
    if os.path.exists(faiss_path) and os.path.exists(meta_path):
        from src.vectorstore import FaissVectorStore
        pdf_store = FaissVectorStore(persist_dir=faiss_pdfs_dir)
        pdf_store.load()
        print(f"[SERVER] PDF index loaded from '{faiss_pdfs_dir}': {len(pdf_store.metadata)} chunks")
    else:
        from src.data_loader import load_all_documents
        docs = load_all_documents(data_dir)
        if docs:
            from src.vectorstore import FaissVectorStore
            pdf_store = FaissVectorStore(persist_dir=faiss_pdfs_dir)
            pdf_store.build_from_documents(docs)
            print(f"[SERVER] PDF index built from '{data_dir}': {len(pdf_store.metadata)} chunks")
        else:
            print(f"[SERVER] No documents in '{data_dir}' — PDF search disabled")


def main():
    global NODE_ID, THUMBNAIL_MAX_SIZE

    parser = argparse.ArgumentParser(description="RAG Federation Server")
    parser.add_argument("--config", type=str, default="peers.json", help="Path to config file")
    parser.add_argument("--port", type=int, default=None, help="Override port from config")
    args = parser.parse_args()

    config = load_config(args.config)
    NODE_ID = config.get("node_id", "unknown")
    THUMBNAIL_MAX_SIZE = tuple(config.get("thumbnail_max_size", [512, 512]))
    port = args.port or config.get("port", 5000)

    print(f"[SERVER] Node '{NODE_ID}' on port {port}")
    print(f"[SERVER] Data: {config.get('data_dir', 'data')}")
    print(f"[SERVER] PDF index: {config.get('faiss_store_dir', 'faiss_store')}")
    print(f"[SERVER] Image index: {config.get('faiss_store_images_dir', 'faiss_store_images')}")
    init_stores(config)

    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
