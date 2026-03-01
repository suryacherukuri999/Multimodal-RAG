import os
import json
import base64
import tempfile
from dotenv import load_dotenv
from pathlib import Path
from src.vectorstore import FaissVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()


def load_node_config(config_path=None):
    """Load config if provided, else return defaults."""
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {}


class RAGSearch:
    def __init__(self, config_path: str = None, federated: bool = False,
                 embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "gpt-4o-mini"):
        config = load_node_config(config_path)

        data_dir = config.get("data_dir", "data")
        persist_dir = config.get("faiss_store_dir", "faiss_store")

        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from src.data_loader import load_all_documents
            docs = load_all_documents(data_dir)
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
        self.llm = ChatOpenAI(model=llm_model)
        print(f"[INFO] OpenAI LLM initialized: {llm_model}")

        # Federation
        self.federated = federated
        self.federation = None
        if federated:
            from src.federation import FederatedSearch
            self.federation = FederatedSearch(config_path=config_path or "peers.json")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)

        if self.federated and self.federation:
            results_merged = self.federation.search_pdfs(results, query, top_k=top_k)
            texts = [r.get("text", "") for r in results_merged if r.get("text")]
        else:
            texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]

        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."

        prompt = (
            f"Answer the following question using only the context provided. "
            f"Be direct and concise.\n\n"
            f"Question: {query}\n\nContext:\n{context}\n\nAnswer:"
        )
        response = self.llm.invoke([prompt])
        return response.content


## surya - CLIP-based Image Search with LLM description + Federation
from src.clip_store import CLIPImageStore


class ImageRAGSearch:
    def __init__(self, config_path: str = None, federated: bool = False):
        config = load_node_config(config_path)

        data_dir = config.get("data_dir", "data")
        persist_dir = config.get("faiss_store_images_dir", "faiss_store_images")

        self.clip_store = CLIPImageStore(persist_dir)
        clip_path = os.path.join(persist_dir, "clip.index")
        meta_path = os.path.join(persist_dir, "clip_meta.pkl")
        if os.path.exists(clip_path) and os.path.exists(meta_path):
            self.clip_store.load()
        else:
            from src.data_loader import load_images
            image_paths = load_images(data_dir)
            if not image_paths:
                print(f"[WARN] No images found in '{data_dir}' folder.")
            else:
                self.clip_store.build_from_images(image_paths)
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        print("[INFO] Image RAG initialized (CLIP + LLM)")

        # Federation
        self.federated = federated
        self.federation = None
        if federated:
            from src.federation import FederatedSearch
            self.federation = FederatedSearch(config_path=config_path or "peers.json")

    def describe_image_from_path(self, image_path: str, query: str) -> str:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = Path(image_path).suffix.lower()
        media_type = "image/png" if ext == ".png" else "image/jpeg"
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Describe this image in detail."},
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
            ]
        )
        response = self.llm.invoke([message])
        return response.content

    def describe_image_from_b64(self, image_b64: str, query: str) -> str:
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Describe this image in detail."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ]
        )
        response = self.llm.invoke([message])
        return response.content

    def _save_remote_image(self, image_b64: str, filename: str) -> str:
        tmp_dir = os.path.join(tempfile.gettempdir(), "rag_federation")
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_path = os.path.join(tmp_dir, filename)
        with open(tmp_path, "wb") as f:
            f.write(base64.b64decode(image_b64))
        return tmp_path

    def search_and_summarize(self, query: str, top_k: int = 3) -> dict:
        local_results = self.clip_store.query_text(query, top_k=top_k)
        for r in local_results:
            print(f"[DEBUG] LOCAL  score={r['score']:.4f} path={r['metadata']['image_path']}")

        if self.federated and self.federation:
            # ── Federated: merge local + remote by score ──
            merged = self.federation.search_images(local_results, query, top_k=top_k)
            images = []
            seen = set()

            for r in merged:
                fname = r["filename"]
                if fname in seen:
                    continue
                seen.add(fname)

                if r["source"] == "local":
                    path = r["image_path"]
                    description = self.describe_image_from_path(path, query)
                    images.append({
                        "path": path,
                        "score": round(r["score"], 4),
                        "description": description,
                        "node_id": r["node_id"],
                        "source": "local"
                    })
                elif r["source"] == "remote" and "image_base64" in r:
                    tmp_path = self._save_remote_image(r["image_base64"], fname)
                    description = self.describe_image_from_b64(r["image_base64"], query)
                    images.append({
                        "path": tmp_path,
                        "score": round(r["score"], 4),
                        "description": description,
                        "node_id": r["node_id"],
                        "source": "remote"
                    })
                else:
                    print(f"[WARN] Skipping '{fname}' from {r['node_id']} — image not fetched")

            if not images and local_results:
                best = local_results[0]
                path = best["metadata"]["image_path"]
                description = self.describe_image_from_path(path, query)
                images.append({
                    "path": path,
                    "score": round(best["score"], 4),
                    "description": description,
                    "node_id": self.federation.node_id,
                    "source": "local"
                })

            return {"images": images}

        else:
            # ── Local only (original behavior) ──
            if not local_results:
                return {"images": []}

            seen = set()
            images = []

            for r in local_results:
                path = r["metadata"]["image_path"]
                filename = Path(path).stem.lower()
                filename_words = set(filename.replace("_", " ").replace("-", " ").split())
                query_words = set(query.lower().split())
                stop_words = {"a", "an", "the", "in", "of", "on", "to", "me", "my", "is", "it",
                              "give", "show", "get"}
                query_words = query_words - stop_words
                name_match = bool(query_words & filename_words)
                if name_match and path not in seen:
                    seen.add(path)
                    description = self.describe_image_from_path(path, query)
                    images.append({
                        "path": path,
                        "score": round(r["score"], 4),
                        "description": description
                    })

            if not images:
                best = local_results[0]
                path = best["metadata"]["image_path"]
                description = self.describe_image_from_path(path, query)
                images.append({
                    "path": path,
                    "score": round(best["score"], 4),
                    "description": description
                })

            return {"images": images}


if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is attention mechanism?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)