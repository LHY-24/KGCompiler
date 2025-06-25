"""
KGCompiler - Operator Fuser Module
Implements the fusion optimizations from:
- Section 4.4 (Operator Fusion Strategies)
- Algorithm 1 (Operator Fusion Algorithm)
"""

from dataclasses import dataclass
from typing import Dict, List, Set
import torch.nn as nn
import torch

@dataclass
class FusedOperator:
    """
    Represents a fused operator group (Def. 4.1 in paper)
    
    Attributes:
        fused_nodes: Original node IDs included in this fusion
        op_type: Type of fused operation ('FUSED_MLP', 'FUSED_INTERSECT', etc.)
        implementation: PyTorch module implementing the fused operation
        input_vars: Input variable node IDs
        output_vars: Output variable node IDs
        fusion_strategy: One of {'HORIZONTAL', 'VERTICAL', 'HYBRID'}
    """
    fused_nodes: List[str]
    op_type: str
    implementation: nn.Module
    input_vars: List[str]
    output_vars: List[str]
    fusion_strategy: str

class OperatorFuser:
    def __init__(self):
        """
        Initializes fusion strategies with thresholds from paper Sec 5.1:
        - Horizontal (sequential ops in single path)
        - Vertical (parallel ops across branches)
        - Hybrid (combination of both)
        """
        self.fusion_strategies = {
            'HORIZONTAL': self._fuse_horizontal,
            'VERTICAL': self._fuse_vertical,
            'HYBRID': self._fuse_hybrid
        }
        self.MAX_FUSION_OPS = 8  # Prevent kernel bloat
        self.MIN_MEM_SAVINGS = 0.2  # At least 20% memory reduction required

    def fuse(self, graph: Dict, patterns: List) -> Dict:
        """
        Main fusion pipeline implementing Algorithm 1:
        1. Apply fusion strategies per pattern
        2. Validate fused operators
        3. Update computation graph
        
        Args:
            graph: Original computation graph from GraphCapturer
            patterns: Recognized patterns from PatternRecognizer
            
        Returns:
            Optimized computation graph with fused operators
            
        Paper Reference:
            "Operator Fuser replaces original operators with fused versions
            while maintaining external dependencies" (Algorithm 1 Step 3)
        """
        fused_graph = graph.copy()
        fused_ops = []

        # Phase 1: Apply fusion strategies
        for pattern in patterns:
            if pattern.fusion_strategy in self.fusion_strategies:
                fused_op = self.fusion_strategies[pattern.fusion_strategy](
                    fused_graph, pattern
                )
                if self._validate_fusion(fused_op):
                    fused_ops.append(fused_op)

        # Phase 2: Update graph
        for fused_op in fused_ops:
            fused_graph = self._update_graph(fused_graph, fused_op)

        # Phase 3: Memory compaction (Appendix C)
        fused_graph = self._compact_memory(fused_graph)

        return fused_graph

    def _fuse_horizontal(self, graph: Dict, pattern) -> FusedOperator:
        """
        Horizontal fusion for sequential operators (paper Sec 4.4):
        - Merges consecutive projections (p→p→p) in tasks like 2p, 3p
        - Fuses MLP chains into single kernel
        
        Paper Reference:
            "Eliminates redundant intermediate steps in single query path"
        """
        nodes = [graph[nid] for nid in pattern.node_ids]
        
        # Case 1: MLP chains for projections (BetaE)
        if pattern.pattern_type == 'MLP_CHAIN':
            mlp_layers = []
            input_dim = None
            
            for node in nodes:
                if node.metadata['impl'] == 'MLP':
                    layers = self._extract_mlp_layers(node)
                    mlp_layers.extend(layers)
                    if input_dim is None:
                        input_dim = node.metadata['input_dim']
            
            # Create fused MLP (preserving dimensions)
            fused_mlp = nn.Sequential(
                nn.Linear(input_dim, mlp_layers[0].out_features),
                *[layer for layer in mlp_layers if not isinstance(layer, nn.Linear)]
            )
            
            return FusedOperator(
                fused_nodes=pattern.node_ids,
                op_type='FUSED_MLP',
                implementation=fused_mlp,
                input_vars=[nodes[0].dependencies[0]],
                output_vars=self._find_outputs(graph, nodes[-1].node_id),
                fusion_strategy='HORIZONTAL'
            )

        # Case 2: Consecutive negations (¬¬)
        elif pattern.pattern_type == 'NEG_CHAIN':
            return FusedOperator(
                fused_nodes=pattern.node_ids,
                op_type='FUSED_NEG',
                implementation=nn.Identity(),  # Double negation cancels out
                input_vars=[nodes[0].dependencies[0]],
                output_vars=self._find_outputs(graph, nodes[-1].node_id),
                fusion_strategy='HORIZONTAL'
            )

    def _fuse_vertical(self, graph: Dict, pattern) -> FusedOperator:
        """
        Vertical fusion for parallel branches (paper Sec 4.4):
        - Merges intersection (∧) inputs in tasks like 2i, 3i
        - Optimizes memory access across paths
        
        Paper Reference:
            "Reduces redundant intermediate outputs across query branches"
        """
        nodes = [graph[nid] for nid in pattern.node_ids]
        
        # Case 1: BetaE intersection
        if pattern.pattern_type == 'BETA_INTERSECT':
            input_vars = list(set(
                dep for node in nodes 
                for dep in node.dependencies
                if graph[dep].node_type == 'var'
            ))
            
            return FusedOperator(
                fused_nodes=pattern.node_ids,
                op_type='FUSED_INTERSECT',
                implementation=BetaIntersection(),
                input_vars=input_vars,
                output_vars=self._find_outputs(graph, nodes[-1].node_id),
                fusion_strategy='VERTICAL'
            )

        # Case 2: Union branches (∨)
        elif pattern.pattern_type == 'UNION_BRANCHES':
            return FusedOperator(
                fused_nodes=pattern.node_ids,
                op_type='FUSED_UNION',
                implementation=BetaUnion(),
                input_vars=[nodes[0].dependencies[0], nodes[1].dependencies[0]],
                output_vars=self._find_outputs(graph, nodes[-1].node_id),
                fusion_strategy='VERTICAL'
            )

    def _fuse_hybrid(self, graph: Dict, pattern) -> FusedOperator:
        """
        Hybrid fusion for complex patterns (paper Sec 4.4):
        - Handles tasks like pi, ip, 2in with combined strategies
        - Balances local and global optimization
        
        Paper Reference:
            "Achieves balance between path-local and cross-path optimization"
        """
        nodes = [graph[nid] for nid in pattern.node_ids]
        
        # Case 1: Projection-Intersection (pi)
        if (len(nodes) >= 2 and 
            nodes[0].metadata.get('impl') == 'MLP' and 
            nodes[1].metadata.get('impl') == 'BetaIntersection'):
            
            mlp_layers = self._extract_mlp_layers(nodes[0])
            intersect = BetaIntersection()
            
            fused_module = nn.Sequential(
                nn.Sequential(*mlp_layers),
                intersect
            )
            
            return FusedOperator(
                fused_nodes=[nodes[0].node_id, nodes[1].node_id],
                op_type='FUSED_PROJ_INTERSECT',
                implementation=fused_module,
                input_vars=nodes[0].dependencies,
                output_vars=self._find_outputs(graph, nodes[1].node_id),
                fusion_strategy='HYBRID'
            )

    def _extract_mlp_layers(self, node: Dict) -> List[nn.Module]:
        """Reconstructs MLP layers from node metadata (paper Eq 4)"""
        layers = []
        if node.metadata['impl'] == 'MLP':
            layers.append(nn.Linear(
                node.metadata['input_dim'],
                node.metadata['hidden_dim']
            ))
            layers.append(nn.ReLU())
            if 'output_dim' in node.metadata:
                layers.append(nn.Linear(
                    node.metadata['hidden_dim'],
                    node.metadata['output_dim']
                ))
        return layers

    def _validate_fusion(self, fused_op: FusedOperator) -> bool:
        """Validates fusion meets paper's criteria (Sec 5.1)"""
        num_ops = len(fused_op.fused_nodes)
        mem_saving = self._estimate_memory_saving(fused_op)
        return (num_ops <= self.MAX_FUSION_OPS and 
                mem_saving >= self.MIN_MEM_SAVINGS)

    def _update_graph(self, graph: Dict, fused_op: FusedOperator) -> Dict:
        """Replaces original nodes with fused operator"""
        new_graph = {
            nid: node for nid, node in graph.items()
            if nid not in fused_op.fused_nodes
        }
        
        # Add fused node
        fused_id = f"fused_{len(new_graph)}"
        new_graph[fused_id] = ComputationNode(
            node_type="op",
            name=fused_op.op_type,
            dependencies=fused_op.input_vars,
            metadata={
                "impl": "fused",
                "module": fused_op.implementation,
                "strategy": fused_op.fusion_strategy
            }
        )
        
        # Rewire outputs
        for out_var in fused_op.output_vars:
            if out_var in new_graph:
                new_graph[out_var].dependencies = [
                    fused_id if dep in fused_op.fused_nodes else dep
                    for dep in new_graph[out_var].dependencies
                ]
        
        return new_graph

    def _compact_memory(self, graph: Dict) -> Dict:
        """Applies memory optimizations from Appendix C"""
        # Implementation details omitted for brevity
        return graph

class BetaIntersection(nn.Module):
    """Implements BetaE intersection operator (Ren & Leskovec 2020b)"""
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        # alpha_out = sum(alphas) - (n-1)
        # beta_out = sum(betas) - (n-1)
        alphas = torch.stack([x[:, 0] for x in inputs])
        betas = torch.stack([x[:, 1] for x in inputs])
        return torch.stack([
            alphas.sum(dim=0) - (len(inputs) - 1),
            betas.sum(dim=0) - (len(inputs) - 1)
        ], dim=1)

class BetaUnion(nn.Module):
    """Implements BetaE union operator (1 - ∏(1 - p_i))"""
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        # Implementation omitted
        pass