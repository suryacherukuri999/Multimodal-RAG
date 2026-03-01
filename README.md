# RAG-DistressNet

A Retrieval-Augmented Generation (RAG) system that supports searching through **PDFs** and **Images** using vector embeddings and LLM-powered answers — with **federated search** across multiple nodes.

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
└── dog.jpeg
```

**Supported formats:** PDF, TXT, CSV, Excel (.xlsx), Word (.docx), PNG, JPG, JPEG

## Usage

### Local Search (Single Node)

```bash
# Search PDFs
python app.py --pdfs --query "What is attention mechanism?"

# Search images
python app.py --images --query "Show me the cat"
```

### Federated Search (Multiple Nodes)

#### Testing on a Single Machine

```bash
# 1. Create separate data folders
bash setup_test.sh

# 2. Put DIFFERENT files in each
cp dog.jpg paper1.pdf data_a/
cp bird.png paper2.pdf data_b/

# 3. Start both servers (separate terminals)
python -m src.server --config peers_a.json   # port 5000, reads data_a/
python -m src.server --config peers_b.json   # port 5001, reads data_b/

# 4. Check connectivity
python app.py --config peers_a.json --discover

# 5. Federated search
python app.py --config peers_a.json --images --federated --query "bird flying"
python app.py --config peers_a.json --pdfs --federated --query "attention mechanism"
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
├── app.py                  # CLI entry point (--pdfs / --images / --federated / --discover)
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
    ├── data_loader.py      # Loads PDFs, TXT, CSV, Excel, Word + image paths
    ├── embedding.py        # Text chunking and embedding (all-MiniLM-L6-v2)
    └── vectorstore.py      # FAISS vector store for text documents
```

## How Federation Works

### Why Scores Are Comparable

All nodes use identical models — `ViT-L-14` for images and `all-MiniLM-L6-v2` for text. Since FAISS uses the same distance metric everywhere (`IndexFlatIP` for CLIP cosine similarity, `IndexFlatL2` for text), scores from different nodes are directly comparable and can be merged by simple sorting.

### Two-Phase Image Search

Transferring full images over HTTP is expensive. The two-phase approach minimizes bandwidth:

1. **Phase 1:** Send only query text → get back `(score, filename)` per result (~100 bytes each)
2. **Phase 2:** Only fetch the actual image from the node(s) that won the score ranking

This means if your local node has the best match, no images are transferred at all.

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
| `/search/pdfs` | POST | Return scores + text chunks |
| `/health` | GET | Node status + index info |

## Models Used

| Component | Model | Purpose |
|-----------|-------|---------|
| Text Embeddings | `all-MiniLM-L6-v2` (~90MB) | Embeds PDF text chunks |
| Image Embeddings | `ViT-L-14` via OpenCLIP (~900MB) | Embeds images and query text |
| LLM | `gpt-4o-mini` (OpenAI API) | Generates answers from retrieved context |

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

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE) for details.