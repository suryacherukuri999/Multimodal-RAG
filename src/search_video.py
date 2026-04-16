## surya - Video RAG Search: keyframe extraction + CLIP retrieval + multi-frame LLM reasoning + federation
import os
import json
import base64
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from src.clip_store_video import CLIPVideoStore
from src.video_extractor_video import extract_all_videos
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()


def load_node_config(config_path=None):
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {}


class VideoRAGSearch:
    def __init__(self, config_path=None, federated=False, data_dir="data",
                 persist_dir="faiss_store_videos", frames_dir="video_frames",
                 scene_threshold=0.4, min_interval_sec=3.0,
                 max_frames_per_video=50, dedup_threshold=0.98, llm_model="gpt-4o"):

        config = load_node_config(config_path)
        data_dir = config.get("data_dir", data_dir)
        persist_dir = config.get("faiss_store_videos_dir", persist_dir)
        frames_dir = config.get("video_frames_dir", frames_dir)

        self.data_dir = data_dir
        self.persist_dir = persist_dir
        self.frames_dir = frames_dir

        self.clip_store = CLIPVideoStore(persist_dir)

        # Check if index already exists
        index_path = os.path.join(persist_dir, "clip_video.index")
        meta_path = os.path.join(persist_dir, "clip_video_meta.pkl")

        if os.path.exists(index_path) and os.path.exists(meta_path):
            self.clip_store.load()
        else:
            print("[INFO] No existing video index found. Extracting keyframes...")
            frame_metadata = extract_all_videos(
                data_dir,
                output_dir=frames_dir,
                scene_threshold=scene_threshold,
                min_interval_sec=min_interval_sec,
                max_frames_per_video=max_frames_per_video
            )
            if frame_metadata:
                self.clip_store.build_from_frames(frame_metadata, dedup_threshold=dedup_threshold)
            else:
                print("[WARN] No video frames extracted. Video search will return empty results.")

        self.llm = ChatOpenAI(model=llm_model)
        print(f"[INFO] Video RAG initialized (CLIP + LLM: {llm_model})")

        # Federation
        self.federated = federated
        self.federation = None
        if federated:
            from src.federation import FederatedSearch
            self.federation = FederatedSearch(config_path=config_path or "peers.json")

    def _encode_frame_b64(self, image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _format_timestamp(self, seconds):
        mins = int(seconds) // 60
        secs = int(seconds) % 60
        return f"{mins}:{secs:02d}"

    def _save_remote_frame(self, image_b64, filename):
        tmp_dir = os.path.join(tempfile.gettempdir(), "rag_federation_videos")
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_path = os.path.join(tmp_dir, filename)
        with open(tmp_path, "wb") as f:
            f.write(base64.b64decode(image_b64))
        return tmp_path

    def _select_frames_for_llm(self, frames_info, score_ratio=0.75, max_frames=12):
        """Smart frame selection: include all frames with scores within score_ratio of the best.
        This adapts to the query — relevant queries send more frames, narrow queries send fewer.

        Args:
            frames_info: list of frame dicts with 'score' key, sorted by score descending
            score_ratio: include frames with score >= best_score * score_ratio
            max_frames: hard cap to prevent token explosion
        """
        if not frames_info:
            return []

        best_score = frames_info[0]["score"]
        if best_score <= 0:
            return frames_info[:max_frames]

        threshold = best_score * score_ratio
        selected = [f for f in frames_info if f["score"] >= threshold]
        selected = selected[:max_frames]

        print(f"[INFO] Smart frame selection: {len(selected)}/{len(frames_info)} frames "
              f"(best={best_score:.4f}, threshold={threshold:.4f})")
        return selected

    def search_and_summarize(self, query, top_k=10, score_ratio=0.75, max_frames=12):
        """Search video frames across local + federated nodes and use LLM for multi-frame reasoning.

        Args:
            query: Natural language query (e.g. "is there a red car?", "how many injured persons?")
            top_k: Number of frames to retrieve from CLIP (initial retrieval pool)
            score_ratio: Include frames with score >= best_score * ratio (smart filtering)
            max_frames: Hard cap on frames sent to LLM

        Returns:
            dict with "answer", "frames" (matched frames with timestamps + node info), "videos"
        """
        if self.clip_store.index is None or self.clip_store.index.ntotal == 0:
            local_results = []
        else:
            local_results = self.clip_store.query_text(query, top_k=top_k)

        if self.federated and self.federation:
            # ── Federated: merge local + remote by score ──
            merged = self.federation.search_videos(local_results, query, top_k=top_k)
            frames_info = []
            seen = set()

            for r in merged:
                fname = r["filename"]
                if fname in seen:
                    continue
                seen.add(fname)

                frame = {
                    "score": round(r["score"], 4),
                    "timestamp": self._format_timestamp(r["timestamp_sec"]),
                    "timestamp_sec": r["timestamp_sec"],
                    "video_name": r["video_name"],
                    "frame_index": r["frame_index"],
                    "node_id": r["node_id"],
                    "source": r["source"]
                }

                if r["source"] == "local":
                    frame["path"] = r["image_path"]
                    frame["video_path"] = r["video_path"]
                    frames_info.append(frame)
                elif r["source"] == "remote" and "image_base64" in r:
                    tmp_path = self._save_remote_frame(r["image_base64"], fname)
                    frame["path"] = tmp_path
                    frame["video_path"] = ""
                    frame["image_b64"] = r["image_base64"]
                    frames_info.append(frame)
                else:
                    print(f"[WARN] Skipping frame '{fname}' from {r['node_id']} — image not fetched")

            # Fallback: if federation returned nothing but we have local results
            if not frames_info and local_results:
                for r in local_results[:top_k]:
                    meta = r["metadata"]
                    frames_info.append({
                        "path": meta["image_path"],
                        "score": round(r["score"], 4),
                        "timestamp": self._format_timestamp(meta["timestamp_sec"]),
                        "timestamp_sec": meta["timestamp_sec"],
                        "video_name": meta["video_name"],
                        "video_path": meta["video_path"],
                        "frame_index": meta["frame_index"],
                        "node_id": self.federation.node_id,
                        "source": "local"
                    })

        else:
            # ── Local only ──
            if not local_results:
                return {"answer": "No relevant video frames found.", "frames": [], "videos": []}

            frames_info = []
            for r in local_results:
                meta = r["metadata"]
                frames_info.append({
                    "path": meta["image_path"],
                    "score": round(r["score"], 4),
                    "timestamp": self._format_timestamp(meta["timestamp_sec"]),
                    "timestamp_sec": meta["timestamp_sec"],
                    "video_name": meta["video_name"],
                    "video_path": meta["video_path"],
                    "frame_index": meta["frame_index"],
                    "source": "local"
                })

        if not frames_info:
            return {"answer": "No relevant video frames found.", "frames": [], "videos": []}

        # Identify unique source videos
        videos = list({f.get("video_path", "") for f in frames_info if f.get("video_path")})

        # ── Smart frame selection for LLM ──
        frames_for_llm = self._select_frames_for_llm(frames_info, score_ratio, max_frames)

        # Build node context for the prompt
        node_info_str = ""
        if self.federated:
            nodes_involved = list({f.get("node_id", "local") for f in frames_for_llm})
            node_info_str = (
                f"\nThese frames come from {len(nodes_involved)} node(s): {', '.join(nodes_involved)}. "
                f"Consider ALL frames from ALL nodes when answering.\n"
            )

        content_parts = [{
            "type": "text",
            "text": (
                f"You are analyzing frames extracted from video(s). "
                f"Each frame has a timestamp showing when it appears in the video.\n"
                f"{node_info_str}\n"
                f"Frames provided (in chronological order):\n"
                + "\n".join(
                    f"- Frame at {f['timestamp']} from '{f['video_name']}'"
                    + (f" [node: {f['node_id']}]" if f.get("node_id") else "")
                    for f in frames_for_llm
                )
                + f"\n\nBased on these video frames, answer the following question. "
                f"Be specific and reference timestamps when relevant. "
                f"Look carefully at every frame.\n\n"
                f"Question: {query}"
            )
        }]

        # Add frame images
        for f in frames_for_llm:
            try:
                if "image_b64" in f:
                    b64 = f["image_b64"]
                else:
                    b64 = self._encode_frame_b64(f["path"])
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                })
            except Exception as e:
                print(f"[ERROR] Failed to encode frame {f['path']}: {e}")

        message = HumanMessage(content=content_parts)
        response = self.llm.invoke([message])

        return {
            "answer": response.content,
            "frames": frames_info,
            "videos": videos
        }


if __name__ == "__main__":
    search = VideoRAGSearch(data_dir="data")
    result = search.search_and_summarize("What is happening in the video?")
    print("Answer:", result["answer"])
    for f in result["frames"]:
        print(f"  [{f['timestamp']}] {f['video_name']} (score={f['score']})")
## surya end
