"""
src/federation.py — Federated search across multiple RAG nodes.

Two-phase image search:
    Phase 1: Fan out query to all peers → collect (score, filename, node_id) — lightweight
    Phase 2: Fetch actual image only from the winning node — single heavy transfer

Single-phase PDF search:
    Fan out query to all peers → collect (score, text, source, node_id) → merge and rank
"""

import os
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional


class FederatedSearch:
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = "peers.json"
        self.config = self._load_config(config_path)
        self.node_id = self.config.get("node_id", "local")
        self.peers = self.config.get("peers", [])
        self.timeout = self.config.get("timeout_seconds", 3)
        print(f"[FEDERATION] Node '{self.node_id}' initialized with {len(self.peers)} peer(s)")
        if self.peers:
            print(f"[FEDERATION] Peers: {self.peers}")

    def _load_config(self, config_path: str) -> dict:
        if not os.path.exists(config_path):
            print(f"[FEDERATION] Config '{config_path}' not found — running in local-only mode")
            return {"node_id": "local", "peers": [], "timeout_seconds": 3}
        with open(config_path, "r") as f:
            return json.load(f)

    # ── Peer Communication ─────────────────────────────────────────────────

    def _query_peer_image_scores(self, peer_url: str, query: str, top_k: int) -> List[Dict]:
        """Phase 1: Get scores from a peer (lightweight)."""
        try:
            resp = requests.post(
                f"{peer_url}/search/images/scores",
                json={"query": query, "top_k": top_k},
                timeout=self.timeout
            )
            if resp.status_code == 200:
                return resp.json()
            else:
                print(f"[FEDERATION] Peer {peer_url} returned {resp.status_code}: {resp.text}")
                return []
        except requests.exceptions.Timeout:
            print(f"[FEDERATION] Peer {peer_url} timed out")
            return []
        except requests.exceptions.ConnectionError:
            print(f"[FEDERATION] Peer {peer_url} unreachable")
            return []
        except Exception as e:
            print(f"[FEDERATION] Error querying peer {peer_url}: {e}")
            return []

    def _fetch_peer_image(self, peer_url: str, filename: str) -> Optional[Dict]:
        """Phase 2: Fetch actual image from the winning peer."""
        try:
            resp = requests.post(
                f"{peer_url}/search/images/fetch",
                json={"filename": filename},
                timeout=self.timeout + 5
            )
            if resp.status_code == 200:
                return resp.json()
            else:
                print(f"[FEDERATION] Failed to fetch image from {peer_url}: {resp.status_code}")
                return None
        except Exception as e:
            print(f"[FEDERATION] Error fetching image from {peer_url}: {e}")
            return None

    def _query_peer_pdfs(self, peer_url: str, query: str, top_k: int) -> List[Dict]:
        """Single phase: Get PDF chunks from a peer."""
        try:
            resp = requests.post(
                f"{peer_url}/search/pdfs",
                json={"query": query, "top_k": top_k},
                timeout=self.timeout
            )
            if resp.status_code == 200:
                return resp.json()
            else:
                print(f"[FEDERATION] Peer {peer_url} returned {resp.status_code}")
                return []
        except requests.exceptions.Timeout:
            print(f"[FEDERATION] Peer {peer_url} timed out")
            return []
        except requests.exceptions.ConnectionError:
            print(f"[FEDERATION] Peer {peer_url} unreachable")
            return []
        except Exception as e:
            print(f"[FEDERATION] Error querying peer {peer_url}: {e}")
            return []

    def _check_peer_health(self, peer_url: str) -> Optional[Dict]:
        """Check if a peer is alive and what indexes it has."""
        try:
            resp = requests.get(f"{peer_url}/health", timeout=2)
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception:
            return None

    # ── Federated Image Search (Two-Phase) ─────────────────────────────────

    def search_images(self, local_results: List[Dict], query: str, top_k: int = 3) -> List[Dict]:
        """
        Federated image search using two-phase approach.

        Args:
            local_results: Results from local CLIP search
                           [{"score": float, "metadata": {"image_path": str}}]
            query: The search query
            top_k: Number of final results to return

        Returns:
            Merged results sorted by score (highest first).
            Each result has: score, filename, node_id, source ("local"/"remote"),
            and either image_path (local) or image_base64 (remote).
        """
        # Normalize local results
        all_scored = []
        for r in local_results:
            path = r["metadata"]["image_path"]
            all_scored.append({
                "score": r["score"],
                "filename": os.path.basename(path),
                "node_id": self.node_id,
                "source": "local",
                "image_path": path,
                "peer_url": None
            })

        if not self.peers:
            all_scored.sort(key=lambda x: x["score"], reverse=True)
            return all_scored[:top_k]

        # ── Phase 1: Collect scores from all peers in parallel ──
        print(f"[FEDERATION] Phase 1: Querying {len(self.peers)} peer(s) for scores...")
        with ThreadPoolExecutor(max_workers=len(self.peers)) as executor:
            future_to_peer = {
                executor.submit(self._query_peer_image_scores, peer, query, top_k): peer
                for peer in self.peers
            }
            for future in as_completed(future_to_peer):
                peer_url = future_to_peer[future]
                try:
                    peer_results = future.result()
                    for pr in peer_results:
                        all_scored.append({
                            "score": pr["score"],
                            "filename": pr["filename"],
                            "node_id": pr.get("node_id", "unknown"),
                            "source": "remote",
                            "image_path": None,
                            "peer_url": peer_url
                        })
                    print(f"[FEDERATION] Got {len(peer_results)} score(s) from {peer_url}")
                except Exception as e:
                    print(f"[FEDERATION] Failed to get scores from {peer_url}: {e}")

        # Sort all by score (CLIP cosine similarity — higher is better)
        all_scored.sort(key=lambda x: x["score"], reverse=True)
        top_results = all_scored[:top_k]

        # ── Phase 2: Fetch actual images for remote winners ──
        remote_winners = [r for r in top_results if r["source"] == "remote"]
        if remote_winners:
            print(f"[FEDERATION] Phase 2: Fetching {len(remote_winners)} image(s) from peer(s)...")
            with ThreadPoolExecutor(max_workers=len(remote_winners)) as executor:
                future_to_result = {
                    executor.submit(
                        self._fetch_peer_image, r["peer_url"], r["filename"]
                    ): r
                    for r in remote_winners
                }
                for future in as_completed(future_to_result):
                    result_entry = future_to_result[future]
                    try:
                        fetched = future.result()
                        if fetched and "image_base64" in fetched:
                            result_entry["image_base64"] = fetched["image_base64"]
                            print(f"[FEDERATION] Fetched '{result_entry['filename']}' from {result_entry['node_id']}")
                        else:
                            print(f"[FEDERATION] Could not fetch '{result_entry['filename']}'")
                    except Exception as e:
                        print(f"[FEDERATION] Fetch failed for '{result_entry['filename']}': {e}")

        # Log final ranking
        for i, r in enumerate(top_results):
            src = f"local ({r['image_path']})" if r["source"] == "local" else f"remote ({r['node_id']})"
            print(f"[FEDERATION] #{i+1} score={r['score']:.4f} file={r['filename']} from={src}")

        return top_results

    # ── Federated PDF Search (Single-Phase) ────────────────────────────────

    def search_pdfs(self, local_results: List[Dict], query: str, top_k: int = 5) -> List[Dict]:
        """
        Federated PDF search — single phase.

        Args:
            local_results: Results from local FAISS text search
                           [{"distance": float, "metadata": {"text": str}}]
            query: The search query
            top_k: Number of final results

        Returns:
            Merged results sorted by distance (lowest first for L2).
        """
        all_results = []
        for r in local_results:
            meta = r.get("metadata", {})
            all_results.append({
                "distance": float(r["distance"]),
                "text": meta.get("text", ""),
                "source": meta.get("source", "local"),
                "node_id": self.node_id
            })

        if not self.peers:
            all_results.sort(key=lambda x: x["distance"])
            return all_results[:top_k]

        print(f"[FEDERATION] Querying {len(self.peers)} peer(s) for PDFs...")
        with ThreadPoolExecutor(max_workers=len(self.peers)) as executor:
            future_to_peer = {
                executor.submit(self._query_peer_pdfs, peer, query, top_k): peer
                for peer in self.peers
            }
            for future in as_completed(future_to_peer):
                peer_url = future_to_peer[future]
                try:
                    peer_results = future.result()
                    for pr in peer_results:
                        all_results.append({
                            "distance": pr["score"],
                            "text": pr.get("text", ""),
                            "source": pr.get("source", peer_url),
                            "node_id": pr.get("node_id", "unknown")
                        })
                    print(f"[FEDERATION] Got {len(peer_results)} chunk(s) from {peer_url}")
                except Exception as e:
                    print(f"[FEDERATION] Failed to get PDFs from {peer_url}: {e}")

        all_results.sort(key=lambda x: x["distance"])
        return all_results[:top_k]

    # ── Network Discovery ──────────────────────────────────────────────────

    def discover_peers(self) -> List[Dict]:
        """Check which peers are alive and what they have."""
        statuses = []
        for peer in self.peers:
            health = self._check_peer_health(peer)
            if health:
                statuses.append({"url": peer, "status": "online", **health})
                print(f"[FEDERATION] ✓ {peer} — {health.get('node_id')} "
                      f"(images: {health.get('image_count', 0)}, pdfs: {health.get('has_pdfs')})")
            else:
                statuses.append({"url": peer, "status": "offline"})
                print(f"[FEDERATION] ✗ {peer} — offline")
        return statuses
