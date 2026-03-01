import argparse
import subprocess


def run_pdfs(query, config_path=None, federated=False):
    from src.search import RAGSearch
    rag_search = RAGSearch(config_path=config_path, federated=federated)
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)


def run_images(query, config_path=None, federated=False):
    from src.search import ImageRAGSearch
    rag_search = ImageRAGSearch(config_path=config_path, federated=federated)
    result = rag_search.search_and_summarize(query, top_k=3)
    if result["images"]:
        print(f"\nResults for: '{query}'")
        for img in result["images"]:
            node_info = ""
            if img.get("source") == "remote":
                node_info = f"  [FROM: {img.get('node_id', 'remote')}]"
            elif img.get("source") == "local":
                node_info = f"  [FROM: {img.get('node_id', 'local')}]"

            print(f"\n  Image: {img['path']}")
            print(f"  Score: {img['score']}{node_info}")
            print(f"  Description: {img['description']}")
            try:
                subprocess.run(["chafa", "--size=40x20", img["path"]], check=False)
            except FileNotFoundError:
                print(f"  (install 'chafa' to view images in terminal)")
    else:
        print("No relevant images found.")


def run_discover(config_path=None):
    from src.federation import FederatedSearch
    fed = FederatedSearch(config_path=config_path or "peers.json")
    print(f"\n{'='*50}")
    print(f"  Node: {fed.node_id}")
    print(f"  Peers configured: {len(fed.peers)}")
    print(f"{'='*50}\n")
    if not fed.peers:
        print("No peers configured. Edit your config file to add peer URLs.")
        return
    statuses = fed.discover_peers()
    print(f"\nSummary: {sum(1 for s in statuses if s['status'] == 'online')}/{len(statuses)} peers online")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAG Search — Local & Federated",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local search (original behavior — no config needed)
  python app.py --pdfs --query "What is attention mechanism?"
  python app.py --images --query "show me the cat"

  # Federated search using a specific node config
  python app.py --config peers_a.json --images --federated --query "laughing dog"
  python app.py --config peers_a.json --pdfs --federated --query "attention mechanism"

  # Check which peers are online
  python app.py --config peers_a.json --discover

  # Start servers (run in separate terminals)
  python -m src.server --config peers_a.json
  python -m src.server --config peers_b.json
        """
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to node config (peers_a.json, peers_b.json, etc.)")
    parser.add_argument("--pdfs", action="store_true", help="Search through PDFs")
    parser.add_argument("--images", action="store_true", help="Search through images")
    parser.add_argument("--query", type=str, default="What is attention mechanism?", help="Search query")
    parser.add_argument("--federated", action="store_true", help="Enable federated search across peers")
    parser.add_argument("--discover", action="store_true", help="Check peer connectivity")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild indexes before searching")
    args = parser.parse_args()

    if args.discover:
        run_discover(args.config)
    elif args.images:
        run_images(args.query, config_path=args.config, federated=args.federated)
    else:
        run_pdfs(args.query, config_path=args.config, federated=args.federated)