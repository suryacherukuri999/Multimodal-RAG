#!/bin/bash
# setup_test.sh — Create separate data folders for testing federation on one machine
#
# Usage: bash setup_test.sh

set -e

echo "Creating data directories..."
mkdir -p data_a data_b

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Directory structure:"
echo "  data_a/  ← Put Node A's files here (PDFs, images)"
echo "  data_b/  ← Put Node B's files here (PDFs, images)"
echo ""
echo "Quick start:"
echo "  1. Add different files to data_a/ and data_b/"
echo "  2. Terminal 1:  python -m src.server --config peers_a.json"
echo "  3. Terminal 2:  python -m src.server --config peers_b.json"
echo "  4. Terminal 3:  python app.py --config peers_a.json --discover"
echo "  5. Terminal 3:  python app.py --config peers_a.json --images --federated --query \"laughing dog\""
echo ""
