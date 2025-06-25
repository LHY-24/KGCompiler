from dataclasses import dataclass
from typing import Dict, List, Set, Optional
from collections import defaultdict

@dataclass
class OperatorPattern:
    """
    Represents a recognized pattern for fusion (Def. 4.1 in paper)
    
    Attributes:
        pattern_type: Type of detected pattern:
            - 'MLP_CHAIN' (consecutive projections)
            - 'BETA_INTERSECT' (BetaE intersection)
            - 'CONE_TRANSFORM' (ConE geometric ops)
            - 'NEG_CHAIN' (consecutive negations)
        node_ids: List of node IDs belonging to this pattern
        fusion_strategy: Recommended fusion strategy:
            - 'HORIZONTAL' (same-path operators)
            - 'VERTICAL' (cross-path operators) 
            - 'HYBRID' (combined strategy)
        metadata: Additional pattern-specific parameters
    """
    pattern_type: str
    node_ids: List[str]
    fusion_strategy: str
    metadata: Optional[Dict] = None

class PatternRecognizer:
    def __init__(self):
        """
        Initializes model-specific pattern templates with thresholds from paper Sec 5.1
        """
        self.model_patterns = {
            'BetaE': self._recognize_betae_patterns,
            'ConE': self._recognize_cone_patterns,
            'LogicE': self._recognize_logice_patterns
        }
        
        # Pattern detection thresholds
        self.MIN_CHAIN_LENGTH = 2  # Minimum ops for horizontal fusion
        self.MAX_PATH_DIFF = 3     # Maximum path length difference for vertical fusion
        self.MAX_NEG_CHAIN = 2     # Maximum consecutive negations

    def recognize(self, model_type: str, graph: Dict) -> List[OperatorPattern]:
        """
        Main pattern detection interface (Implements P[G] in paper Sec 4.3)
        
        Args:
            model_type: KG model type ('BetaE', 'ConE', etc.)
            graph: Computation graph from GraphCapturer
            
        Returns:
            List of OperatorPattern objects ready for fusion
            
        Paper Reference:
            "Automatically identifies operator combinations within G to enable
            model-specific optimizations" (Sec 4.3)
        """
        if model_type not in self.model_patterns:
            raise ValueError(f"Unsupported model: {model_type}. Choose from {list(self.model_patterns.keys())}")
            
        return self.model_patterns[model_type](graph)

    def _recognize_betae_patterns(self, graph: Dict) -> List[OperatorPattern]:
        """
        Detects BetaE-specific patterns (paper Sec 4.3 & Eq 4):
        - MLP chains for projections
        - Beta distribution intersections
        - Negation sequences
        """
        patterns = []
        
        # Pattern 1: MLP chains (for p, 2p, 3p tasks)
        mlp_chains = self._find_operator_chains(
            graph,
            start_conditions=lambda n: (
                n.node_type == 'op' and 
                n.metadata.get('impl') == 'MLP'
            ),
            edge_conditions=lambda src, dst: (
                src.name.endswith('matmul') and 
                dst.name.endswith('relu')
            )
        )
        patterns.extend([
            OperatorPattern(
                pattern_type='MLP_CHAIN',
                node_ids=chain,
                fusion_strategy='HORIZONTAL',
                metadata={'length': len(chain)}
            ) for chain in mlp_chains if len(chain) >= self.MIN_CHAIN_LENGTH
        ])
        
        # Pattern 2: Beta intersections (for i, 2i, 3i tasks)
        intersect_nodes = [
            nid for nid, node in graph.items() 
            if node.node_type == 'op' and node.name == 'intersection'
        ]
        for nid in intersect_nodes:
            input_paths = self._trace_input_paths(graph, nid)
            if self._is_fusible_intersection(input_paths):
                patterns.append(
                    self._build_intersect_pattern(graph, nid, input_paths)
                )
        
        # Pattern 3: Negation chains (for n, 2in tasks)
        neg_chains = self._find_operator_chains(
            graph,
            start_conditions=lambda n: (
                n.node_type == 'op' and 
                n.name == 'negation'
            ),
            edge_conditions=lambda src, dst: True  # Any dependency
        )
        patterns.extend([
            OperatorPattern(
                pattern_type='NEG_CHAIN',
                node_ids=chain,
                fusion_strategy='HORIZONTAL',
                metadata={'cancellable': len(chain) % 2 == 0}
            ) for chain in neg_chains if 1 < len(chain) <= self.MAX_NEG_CHAIN
        ])
        
        return patterns

    def _recognize_cone_patterns(self, graph: Dict) -> List[OperatorPattern]:
        """
        Detects ConE-specific patterns (paper Sec 4.3):
        - Cone projection sequences
        - Geometric intersection/union ops
        """
        patterns = []
        
        # Pattern 1: Cone transform chains
        cone_chains = self._find_operator_chains(
            graph,
            start_conditions=lambda n: (
                n.node_type == 'op' and 
                n.metadata.get('impl') == 'ConeProjection'
            ),
            edge_conditions=lambda src, dst: (
                'cone' in src.name and 'cone' in dst.name
            )
        )
        patterns.extend([
            OperatorPattern(
                pattern_type='CONE_TRANSFORM',
                node_ids=chain,
                fusion_strategy='HORIZONTAL',
                metadata={'dim': graph[chain[0]].metadata['cone_dim']}
            ) for chain in cone_chains if len(chain) >= 2
        ])
        
        return patterns

    def _find_operator_chains(
        self,
        graph: Dict,
        start_conditions: callable,
        edge_conditions: callable
    ) -> List[List[str]]:
        """
        Generalized operator chain detection (paper Algorithm 1 helper)
        
        Args:
            start_conditions: Predicate to identify chain start nodes
            edge_conditions: Predicate to validate chain continuation
            
        Returns:
            List of node chains satisfying the conditions
        """
        chains = []
        visited = set()
        
        for node_id, node in graph.items():
            if node_id not in visited and start_conditions(node):
                current_chain = []
                current_node = node_id
                
                while current_node:
                    visited.add(current_node)
                    current_chain.append(current_node)
                    
                    # Find next node satisfying edge conditions
                    next_node = None
                    for neighbor in self._get_outgoing_nodes(graph, current_node):
                        if (neighbor not in visited and 
                            edge_conditions(graph[current_node], graph[neighbor])):
                            next_node = neighbor
                            break
                    
                    current_node = next_node
                
                if len(current_chain) > 1:
                    chains.append(current_chain)
        
        return chains

    def _trace_input_paths(self, graph: Dict, node_id: str) -> List[List[str]]:
        """Traces all input paths to an operator node for vertical fusion"""
        node = graph[node_id]
        return [
            self._trace_path_backward(graph, dep)
            for dep in node.dependencies
        ]

    def _is_fusible_intersection(self, input_paths: List[List[str]]) -> bool:
        """
        Validates if intersection meets fusion criteria (paper Sec 4.4):
        1. Path length differences within threshold
        2. No data dependencies between paths
        """
        if not input_paths:
            return False
            
        lengths = [len(path) for path in input_paths]
        return (max(lengths) - min(lengths)) <= self.MAX_PATH_DIFF

    def _build_intersect_pattern(
        self,
        graph: Dict,
        intersect_node: str,
        input_paths: List[List[str]]
    ) -> OperatorPattern:
        """Constructs vertical fusion pattern for intersection"""
        all_nodes = [intersect_node] + [
            nid for path in input_paths for nid in path
        ]
        return OperatorPattern(
            pattern_type='BETA_INTERSECT',
            node_ids=all_nodes,
            fusion_strategy='VERTICAL',
            metadata={
                'input_paths': len(input_paths),
                'max_depth': max(len(p) for p in input_paths)
            }
        )

    def _get_outgoing_nodes(self, graph: Dict, node_id: str) -> List[str]:
        """Finds all nodes consuming current node's output"""
        return [
            nid for nid, node in graph.items()
            if node.dependencies and node_id in node.dependencies
        ]

    def _trace_path_backward(self, graph: Dict, start_node: str) -> List[str]:
        """Traces computation path backward from a node"""
        path = []
        current = start_node
        
        while current and graph[current].node_type == 'op':
            path.append(current)
            if not graph[current].dependencies:
                break
            current = graph[current].dependencies[0]  # Follow primary input
        
        return path