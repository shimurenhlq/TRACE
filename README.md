# TRACE: Traversal Retrieval-Augmented Chain of Evidence

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Abstract

Early Long-context Document Visual Question Answering (DocVQA) methods struggle with preserving visual semantics or handling finite context windows. Conversely, recent RAG-based approaches suffer from "semantic gaps" and "structural disconnections" due to passive retrieval mechanisms that ignore logical dependencies. To address these challenges, we introduce TRACE (Traversal Retrieval-Augmented Chain of Evidence). By navigating a Bi-Layered Graph that encodes both physical adjacency and semantic relevance, TRACE transforms retrieval from static matching into adaptive evidence chain construction. Furthermore, we propose M5BookVQA, a benchmark designed to assess deep, multi-hop reasoning in books, addressing the limitations of existing datasets. Extensive experiments show that TRACE achieves an average accuracy improvement of 14.07% on M5BookVQA and exhibits robust generalization with a 13.38% gain across four established benchmarks.

## Architecture

```
User Question
     ↓
┌────────────────┐
│    Planner     │ → Decomposes query into sub-queries
└────────────────┘
     ↓
┌────────────────┐
│   Navigator    │ → Retrieves relevant pages via:
│                │   • ColPali semantic search
│                │   • VLM relevance filtering
│                │   • Bi-Layered Graph traversal
└────────────────┘
     ↓
┌────────────────┐
│    Reasoner    │ → Multi-modal reasoning over evidence
└────────────────┘
     ↓
   Answer
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for ColPali)
- 16GB+ RAM

### Setup

1. Clone the repository:
```bash
git clone https://github.com/shimurenhlq/TRACE.git
cd TRACE
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the ColPali model:
```bash
# Download from Hugging Face or specify your local path
# Model: vidore/colpali-v1.2
```

4. Configure API keys:
```bash
cp config.example.yaml config.yaml
# Edit config.yaml with your API credentials
```

## Quick Start

### 1. Prepare Data

Preprocess your document images and build embeddings:

```bash
# Generate ColPali embeddings and Bi-Layered graphs for each chapter
python preprocessing/prepare_embeddings.py \
  --data_path data/m3bookvqa/data.jsonl \
  --image_root data/images \
  --output_dir data/processed \
  --colpali_model_path vidore/colpali-v1.2

# (Optional) Merge for book-level or global retrieval
python preprocessing/merge_resources.py \
  --input_dir data/processed \
  --image_root data/images \
  --mode book  # or 'global'
```

### 2. Run Inference

**Chapter Mode** (fastest):
```bash
python scripts/run_chapter_mode.py \
  --data_path data/m3bookvqa/data.jsonl \
  --image_root data/images \
  --asset_dir data/processed \
  --output_file results/chapter_results.json
```

**Book Mode** (balanced):
```bash
python scripts/run_book_mode.py \
  --data_path data/m3bookvqa/data.jsonl \
  --image_root data/images \
  --asset_dir data/processed \
  --output_file results/book_results.json
```

**Global Mode** (maximum coverage):
```bash
python scripts/run_global_mode.py \
  --data_path data/m3bookvqa/data.jsonl \
  --image_root data/images \
  --asset_dir data/processed \
  --output_file results/global_results.json
```

### 3. Evaluate Results

```bash
python scripts/evaluate.py \
  --result_file results/chapter_results.json
```

## Configuration

Edit `config.yaml` to customize:

- **Model Endpoints**: Configure Planner, Navigator, and Reasoner models
- **Retrieval Parameters**: Adjust top-k, threshold, graph expansion depth
- **API Keys**: Set your OpenAI/Anthropic/Aliyun credentials

Example configuration:
```yaml
models:
  planner:
    provider: "openai"  # or "aliyun", "anthropic"
    model: "gpt-4"
    api_key: "${OPENAI_API_KEY}"
  
  navigator:
    provider: "aliyun"
    model: "qwen-vl-plus"
    api_key: "${ALIYUN_API_KEY}"
  
  reasoner:
    provider: "openai"
    model: "gpt-4-vision"
    api_key: "${OPENAI_API_KEY}"

retrieval:
  top_k: 10
  graph_threshold: 0.7
  max_pages_per_step: 3
```

## Dataset

The M3BookVQA dataset used in this project is available on Hugging Face:
- **Format**: JSONL with question, options, answer, and page references
- **Content**: Multi-modal questions over book-length documents
- **Download**: [HuggingFace Dataset](https://huggingface.co/datasets/shimurenhlq/M3BookVQA) (Coming soon)

Sample data format:
```json
{
  "id": "chapter_name-1",
  "question": "What is the main topic discussed in this chapter?",
  "options": ["A. Topic 1", "B. Topic 2", "C. Topic 3", "D. Topic 4"],
  "answer": "A",
  "page_numbers": [6, 7, 8],
  "topic": "history"
}
```

## Project Structure

```
TRACE/
├── src/                      # Core modules
│   ├── config.py            # Configuration management
│   ├── environment.py       # BookEnvironment class
│   ├── agents.py            # AgenticSystem class
│   └── prompts.py           # Agent prompts
├── preprocessing/           # Data preprocessing
│   ├── prepare_embeddings.py
│   └── merge_resources.py
├── scripts/                 # Execution scripts
│   ├── run_chapter_mode.py
│   ├── run_book_mode.py
│   ├── run_global_mode.py
│   └── evaluate.py
└── requirements.txt
```

## Citation

If you use this code in your research, please cite our paper (citation will be added upon publication).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- ColPali for multi-vector document embeddings
- AutoGen for agent orchestration

## Contact

For questions or issues, please open an issue on GitHub.
