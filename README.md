# KGCompiler

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)    
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.3+](https://img.shields.io/badge/PyTorch-2.3%2B-orange)](https://pytorch.org/)

Official resources of **"KGCompiler: Deep Learning Compilation Optimization for Knowledge Graph Complex Logical Query Answering"**.

---

## 🔍 Overview

**KGCompiler** (**K**nowledge **G**raph **Compiler**) is the first knowledge graph-oriented deep learning compiler designed to optimize Complex Logical Query Answering (CLQA) tasks. By introducing KG-specific compilation optimizations, it achieves **average 3.71× speedup** and significant memory reduction for state-of-the-art KG models without compromising accuracy.

KGCompiler addresses three key challenges in CLQA:

1. **Semantic Gap Between Logical Operators and Hardware Execution Paradigms**
2. **Dynamic Query Structures Defy Static Optimization**
3. **Tight Coupling of Embedding Methods and Optimization Rules**

Through three core components:
- **Graph Capturer**: Converts KG models to computation graphs
- **Pattern Recognizer**: Detects FOL operator combinations
- **Operator Fuser**: Applies KG-specific fusion strategies

![KGCompiler Architecture](docs/figures/figure3.jpg)

---

## 🚀 Quick Start  

### Models
- [x] [CQD](https://arxiv.org/abs/2011.03459)  
- [x] [BetaE](https://arxiv.org/abs/2010.11465)
- [x] [Query2box](https://arxiv.org/abs/2002.05969)
- [x] [GQE](https://arxiv.org/abs/1806.01445)

### KG Data
The KG data (FB15k, FB15k-237, NELL995) mentioned in the BetaE paper and the Query2box paper can be downloaded [here](http://snap.stanford.edu/betae/KG_data.zip).

### Installation
```bash
git clone https://github.com/LHY-24/KGCompiler.git
cd KGCompiler
pip install -r requirements.txt  
```

### Basic Usage
```python
from src.graph_capturer import GraphCapturer
from src.operator_fuser import OperatorFuser

# 1. Convert FOL query to computation graph  
query = "∃v: Winner(TuringAward, v) ∧ Citizen(Canada, v) ∧ Graduate(v, ?)"
graph = GraphCapturer().capture(query)

# 2. Apply KGCompiler optimizations
optimized_graph = OperatorFuser().fuse(graph)

# 3. Execute on supported models (e.g., BetaE)
from src.models.betae import BetaE
model = BetaE(dataset="fb15k-237")
results = model.execute(optimized_graph)
```

---

## 📊 Performance

### Speedup Comparison (Batch Size = 1)
| Model     | Avg Speedup | Max Speedup |
|-----------|------------|------------|
| BetaE     | 7.40×      | 22.68×     |
| ConE      | 6.19×      | 17.25×     |
| Query2Triple | 1.04×    | 19.58×     |

![Performance Comparison](docs/figures/figure4.jpg)

### Memory Reduction
![Memory Usage](docs/figures/figure6.jpg)

---

## 🧩 Supported Features

### Datasets
- FB15K
- FB15K-237  
- NELL

### CLQA Algorithms
| Algorithm | EPFO | Negation |
|-----------|------|----------|
| GQE       | ✅   | ❌       |
| Q2B       | ✅   | ❌       |
| BetaE     | ✅   | ✅       |
| LogicE    | ✅   | ✅       |
| ConE      | ✅   | ✅       |
| Query2Triple | ✅ | ✅       |

### Query Types
- **EPFO**: `1p`, `2p`, `3p`, `2i`, `3i`, `pi`, `ip`, `2u`, `up`
- **Negation**: `2in`, `3in`, `inp`, `pin`, `pni`

---

## 🛠 Customization

### Add New Fusion Strategy
```python
from src.operator_fuser import FusionStrategy

class CustomFusion(FusionStrategy):
    def match_pattern(self, graph):
        # Implement your pattern detection logic
        pass
    
    def fuse(self, graph):
        # Implement fusion optimization
        return optimized_graph

OperatorFuser.register_strategy(CustomFusion())
```

### Extend to New Model
1. Implement model in `src/models/`
2. Add pattern recognition rules in `src/pattern_recognizer.py`

---
