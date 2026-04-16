# RAG-DistressNet

A Retrieval-Augmented Generation (RAG) system that supports searching through **PDFs**, **Images**, and **Videos** using vector embeddings and LLM-powered answers — with **federated search** across multiple nodes.

## Architecture

### PDF Pipeline
```
PDF → PyPDFLoader extracts text → Chunked → Embedded (all-MiniLM-L6-v2) → FAISS index
Query → Embedded (all-MiniLM-L6-v2) → FAISS retrieves top-k chunks → OpenAI LLM generates answer
```

### Image Pipeline
```
Image → CLIP (ViT-L-14) embeds directly → FAISS index
Query → CLIP embeds text → FAISS retrieves best match → OpenAI LLM describes matched image
```

### Video Pipeline
```
Video → OpenCV scene-change detection + interval sampling → keyframes
Keyframes → CLIP deduplication (cosine sim > 0.98 removed) → CLIP (ViT-L-14) embeds → FAISS index
Query → CLIP embeds text → smart score-based frame selection → frames sent to OpenAI GPT-4o
GPT-4o reasons across frames → answer with timestamps (e.g. "Red car visible at 0:12")
```

**Three-stage extraction:**
1. **Scene-change detection:** Histogram diff between consecutive frames catches major visual shifts
2. **Minimum interval sampling:** Fallback every 3 seconds ensures gradual changes aren't missed
3. **CLIP deduplication:** Near-identical frames (cosine similarity > 0.98) are removed before indexing

**Smart frame selection:** Instead of a fixed top-k, all frames scoring within 75% of the best CLIP score are sent to the LLM (capped at 12). This adapts to the query — a broad query sends more frames, a narrow one sends fewer.

### Federated Pipeline
```
User Query
    │
    ▼
Local Node (Coordinator)
    ├── Search local FAISS index → local results (score, image/chunk)
    ├── POST /search to Peer B → remote results (score, data)
    ├── POST /search to Peer C → remote results (score, data)
    │
    ▼
Merge all results by score → pick top-k globally → LLM generates answer
```

**Image search uses a two-phase approach:**
1. **Phase 1 (lightweight):** Fan out query to all peers → collect `(score, filename, node_id)` only
2. **Phase 2 (targeted):** Fetch actual image only from the winning node(s) → single heavy transfer

**Video search uses the same two-phase approach as images**, with extra metadata (timestamp, video_name). All winning frames from all nodes are sent to the LLM **in a single call** so it can reason globally (e.g. count injured persons across all nodes' videos).

**PDF search uses single-phase:** Fan out → collect `(score, text_chunk)` → merge and rank.

## Prerequisites

- **Python 3.11**
- **Conda** (Miniforge recommended for Mac, Miniconda/Anaconda for Linux)
- **OpenAI API key** with credits loaded
- **chafa** (optional, for terminal image display)

## Setup

### Mac (Apple Silicon M1/M2/M3/M4)

> **Important:** Use [Miniforge](https://github.com/conda-forge/miniforge) (ARM-native conda), not Anaconda (x86). Anaconda limits PyTorch to v2.2.2 on Apple Silicon.

```bash
brew install miniforge
/opt/homebrew/Caskroom/miniforge/base/bin/conda init zsh
# Restart terminal

brew install chafa  # optional
```

### Linux (Ubuntu/Debian)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Follow prompts, restart terminal

sudo apt install chafa  # optional
```

### Common Setup (Mac & Linux)

#### 1. Create Conda Environment

```bash
conda create -n rag python=3.11 -y
conda activate rag
```

#### 2. Install PyTorch (first, separately)

**Mac (Apple Silicon):**
```bash
pip install torch torchvision
```

**Linux (CPU only):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Linux (with NVIDIA GPU):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### 3. Install Dependencies

```bash
pip install "numpy<2"
pip install -r requirements.txt
```

#### 4. Set Up API Key

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-proj-your_actual_key_here
```

#### 5. Add Your Data

Place your files in the `data/` folder (or `data_a/`, `data_b/` for federation):

```
data/
├── paper1.pdf
├── paper2.pdf
├── cat.png
├── dog.jpeg
└── surveillance.mp4
```

**Supported formats:** PDF, TXT, CSV, Excel (.xlsx), Word (.docx), PNG, JPG, JPEG, MP4, AVI, MOV, MKV, WebM

## Usage

### Local Search (Single Node)

```bash
# Search PDFs
python app.py --pdfs --query "What is attention mechanism?"

# Search images
python app.py --images --query "Show me the cat"

# Search videos
python app.py --videos --query "is there a red car?"
python app.py --videos --query "how many injured persons?"
python app.py --videos --query "person wearing white shirt" --data-dir data
```

### Federated Search (Multiple Nodes)

#### Testing on a Single Machine

```bash
# 1. Create separate data folders
bash setup_test.sh

# 2. Put DIFFERENT files in each
cp dog.jpg paper1.pdf camera_a.mp4 data_a/
cp bird.png paper2.pdf camera_b.mp4 data_b/

# 3. Start both servers (separate terminals)
python -m src.server --config peers_a.json   # port 8000, reads data_a/
python -m src.server --config peers_b.json   # port 8001, reads data_b/

# 4. Check connectivity
python app.py --config peers_a.json --discover

# 5. Federated search
python app.py --config peers_a.json --images --federated --query "bird flying"
python app.py --config peers_a.json --pdfs --federated --query "attention mechanism"
python app.py --config peers_a.json --videos --federated --query "how many injured persons?"
python app.py --config peers_a.json --videos --federated --query "is there a red car?"
```

#### Multiple Machines

Edit `peers_a.json` with actual IPs:

```json
{
    "node_id": "node-a",
    "port": 5000,
    "data_dir": "data",
    "faiss_store_dir": "faiss_store",
    "faiss_store_images_dir": "faiss_store_images",
    "faiss_store_videos_dir": "faiss_store_videos",
    "video_frames_dir": "video_frames",
    "peers": [
        "http://192.168.1.10:5000",
        "http://192.168.1.11:5000"
    ],
    "timeout_seconds": 3,
    "thumbnail_max_size": [512, 512]
}
```

**Output shows which node each result came from:**
```
Results for: 'laughing dog'

  Image: /tmp/rag_federation/happy_dog.jpg
  Score: 0.3421  [FROM: node-b]
  Description: A golden retriever with its mouth open...

  Image: data_a/my_dog.jpg
  Score: 0.2918  [FROM: node-a]
  Description: A small poodle playing in a park...
```

## Project Structure

```
RAG-DistressNet/
├── app.py                  # CLI entry point (--pdfs / --images / --videos / --federated / --discover)
├── requirements.txt        # Python dependencies
├── setup_test.sh           # Helper to create data_a/ and data_b/ for local testing
├── peers_a.json            # Node A config (port 5000, data_a/)
├── peers_b.json            # Node B config (port 5001, data_b/)
├── .env                    # OpenAI API key (create this)
└── src/
    ├── __init__.py
    ├── server.py           # Flask API server (run on each node)
    ├── federation.py       # Fan-out + score aggregation logic
    ├── search.py           # RAGSearch + ImageRAGSearch (local & federated)
    ├── clip_store.py       # CLIP-based vector store for images (ViT-L-14)
    ├── clip_store_video.py # CLIP-based vector store for video frames
    ├── video_extractor_video.py  # Hybrid keyframe extraction (scene-change + interval + dedup)
    ├── search_video.py     # VideoRAGSearch — multi-frame LLM reasoning
    ├── data_loader.py      # Loads PDFs, TXT, CSV, Excel, Word + image paths
    ├── embedding.py        # Text chunking and embedding (all-MiniLM-L6-v2)
    └── vectorstore.py      # FAISS vector store for text documents
```

## How Federation Works

### Why Scores Are Comparable

All nodes use identical models — `ViT-L-14` for images and `all-MiniLM-L6-v2` for text. Since FAISS uses the same distance metric everywhere (`IndexFlatIP` for CLIP cosine similarity, `IndexFlatL2` for text), scores from different nodes are directly comparable and can be merged by simple sorting.

### Two-Phase Image & Video Search

Transferring full images/frames over HTTP is expensive. The two-phase approach minimizes bandwidth:

1. **Phase 1:** Send only query text → get back `(score, filename)` per result (~100 bytes each)
2. **Phase 2:** Only fetch the actual image/frame from the node(s) that won the score ranking

This means if your local node has the best match, no images are transferred at all.

For video search, Phase 1 also returns `(video_name, timestamp_sec)` so the coordinator knows the temporal context. All winning frames from all nodes are sent to the LLM in one call for global reasoning (e.g. counting people across all cameras).

### Fault Tolerance

- Peers that timeout or are unreachable are silently skipped
- Configurable timeout per peer (default: 3s)
- Local results always available even if all peers are down
- `--discover` flag lets you check peer status before running queries

## API Endpoints (src/server.py)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/search/images/scores` | POST | Phase 1: return scores + filenames |
| `/search/images/fetch` | POST | Phase 2: return image as base64 |
| `/search/videos/scores` | POST | Phase 1: return scores + frame filenames + timestamps |
| `/search/videos/fetch` | POST | Phase 2: return video frame as base64 |
| `/search/pdfs` | POST | Return scores + text chunks |
| `/health` | GET | Node status + index info |

## Models Used

| Component | Model | Purpose |
|-----------|-------|---------|
| Text Embeddings | `all-MiniLM-L6-v2` (~90MB) | Embeds PDF text chunks |
| Image Embeddings | `ViT-L-14` via OpenCLIP (~900MB) | Embeds images and query text |
| Video Keyframes | OpenCV (scene detection) | Extracts keyframes from videos |
| LLM (PDFs/Images) | `gpt-4o-mini` (OpenAI API) | Generates answers from retrieved context |
| LLM (Videos) | `gpt-4o` (OpenAI API) | Multi-frame visual reasoning for video queries |

## Troubleshooting

### Federation: peer shows "offline"
- Make sure `python -m src.server --config <config>.json` is running on the peer
- Check firewall rules — port must be open
- Verify the URL is reachable: `curl http://<peer-ip>:<port>/health`

### Federation: scores seem inconsistent
- All nodes MUST use the same CLIP model (`ViT-L-14`) and text embedding model (`all-MiniLM-L6-v2`)
- Mixing models makes scores incomparable

### PyTorch version stuck at 2.2.2 (Mac)
Use Miniforge instead of Anaconda:
```bash
brew install miniforge
/opt/homebrew/Caskroom/miniforge/base/bin/conda init zsh
```

### NumPy 2.x compatibility error
```bash
pip install "numpy<2"
```

### Segmentation fault with FAISS (Mac)
```bash
pip uninstall faiss-cpu -y
conda install -c conda-forge faiss-cpu -y
```

### Segmentation fault when loading CLIP + FAISS together
This is an import order issue on macOS. `open_clip` and `torch` must be imported **before** `faiss`. This is already handled in the codebase — if you see this in custom scripts, ensure:
```python
import open_clip
import torch
import faiss  # must come AFTER open_clip/torch
```

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE) for details.