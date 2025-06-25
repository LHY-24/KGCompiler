from dataclasses import dataclass
from typing import Dict, List, Tuple, Union
import re

@dataclass
class ComputationNode:
    """
    Represents a node in the computation graph (Def. 3.3 in paper)
    
    Attributes:
        node_type: 'var' (variable/entity) or 'op' (operator)
        name: Descriptive identifier (e.g., 'projection', 'bound_var')
        dependencies: Input node IDs (edges in the computation graph)
        metadata: Operator-specific parameters including:
            - impl: Implementation type ('MLP', 'BetaIntersection', etc.)
            - relation: For projection operators
            - dtype: Data type ('beta_dist', 'cone_embed', etc.)
    """
    node_type: str
    name: str
    dependencies: List[str] = None
    metadata: Dict = None

class GraphCapturer:
    def __init__(self):
        """
        Initializes the FOL to computation graph mapper with:
        - node_counter: Unique ID generator
        - operator_map: Mapping from FOL operators to handler functions
        """
        self.node_counter = 0
        self.operator_map = {
            '∃': self._capture_projection,
            '∧': self._capture_intersection,
            '∨': self._capture_union,
            '¬': self._capture_negation
        }

    def capture(self, fol_query: str) -> Dict[str, ComputationNode]:
        """
        Converts FOL query to computation graph (Implements M[Q] in Eq. 2)
        
        Args:
            fol_query: First-order logic query string, e.g.:
              "∃v: Winner(TuringAward,v) ∧ Citizen(Canada,v) ∧ Graduate(v,?)"
              
        Returns:
            Dictionary of {node_id: ComputationNode} representing the graph
            
        Paper Reference:
            "Each FOL operator is mapped to functions in set F" (Sec 4.2)
        """
        graph = {}
        ast = self._parse_fol(fol_query)
        
        # Recursively process AST nodes
        for operator, operands in ast:
            if operator in self.operator_map:
                self.operator_map[operator](graph, operands)
            else:
                raise ValueError(f"Unsupported operator: {operator}")
                
        return graph

    def _create_node(self, node_type: str, name: str) -> str:
        """Generates unique node ID following {type}_{counter} convention"""
        node_id = f"{node_type}_{self.node_counter}"
        self.node_counter += 1
        return node_id

    def _parse_fol(self, query: str) -> List[Tuple[str, List]]:
        """
        Parses FOL queries into abstract syntax trees (AST)
        
        Example:
            Input: "∃v: P(v) ∧ Q(v,?)"
            Output: [('∃', ['v', ('∧', [('P','v'), ('Q',['v','?'])])]
        """
        # Remove whitespace and split quantifiers
        query = re.sub(r'\s+', '', query)
        quant_part, body = query.split(':', 1)
        
        # Parse quantifier (e.g., ∃v)
        quantifier = quant_part[0]
        bound_var = quant_part[1:]
        
        # Parse body into nested tuples
        return [(quantifier, [bound_var, self._parse_logical_expr(body)])]
    
    def _parse_logical_expr(self, expr: str) -> Union[str, Tuple]:
        """Recursively parses logical expressions"""
        if '(' not in expr:
            return expr
            
        op = expr[0]
        if op in ['∧', '∨']:
            # Binary operators
            parts = self._split_args(expr[1:])
            return (op, [self._parse_logical_expr(p) for p in parts])
        elif op == '¬':
            # Unary operator
            return (op, [self._parse_logical_expr(expr[1:])])
        else:
            # Atomic relation (e.g., P(v))
            rel, args_str = expr.split('(', 1)
            args = args_str[:-1].split(',')
            return (rel, args)

    def _capture_projection(self, graph: Dict, operands: List):
        """
        Handles existential quantification (∃) - maps to projection operator
        
        Paper Reference:
            "r_k(e_i,v_j) → {(var_i,op_k),(op_k,var_j)}" (Eq 2)
        """
        bound_var = operands[0]
        subquery = operands[1]
        
        # Create variable node for bound variable
        var_node = self._create_node("var", f"bound_{bound_var}")
        graph[var_node] = ComputationNode(
            node_type="var",
            name=f"bound_{bound_var}",
            metadata={"dtype": "beta_dist"}
        )
        
        # Process subquery (recursive)
        if isinstance(subquery, tuple):
            op_node = self._create_node("op", "projection")
            self.operator_map[subquery[0]](graph, subquery[1])
            graph[op_node] = ComputationNode(
                node_type="op",
                name="projection",
                dependencies=[var_node, self._get_last_op_node(graph)],
                metadata={
                    "impl": "MLP",
                    "relation": subquery[0] if isinstance(subquery[0], str) else None
                }
            )

    def _capture_intersection(self, graph: Dict, operands: List):
        """
        Handles conjunction (∧) - maps to intersection operator
        
        Paper Reference:
            "∧(Q1,Q2) → f_and(M[Q1],M[Q2])" (Eq 2)
        """
        op_node = self._create_node("op", "intersection")
        input_nodes = []
        
        for operand in operands:
            if isinstance(operand, tuple):
                self.operator_map[operand[0]](graph, operand[1])
                input_nodes.append(self._get_last_op_node(graph))
            else:
                # Handle atomic relations
                rel_node = self._create_node("op", f"relation_{operand}")
                graph[rel_node] = ComputationNode(
                    node_type="op",
                    name=f"relation_{operand}",
                    metadata={"impl": "Lookup"}
                )
                input_nodes.append(rel_node)
        
        graph[op_node] = ComputationNode(
            node_type="op",
            name="intersection",
            dependencies=input_nodes,
            metadata={
                "impl": "BetaIntersection",
                "method": "product_of_betas"  # From BetaE paper
            }
        )

    def _get_last_op_node(self, graph: Dict) -> str:
        """Helper to find the most recently added operator node"""
        for nid in reversed(graph.keys()):
            if graph[nid].node_type == 'op':
                return nid
        raise ValueError("No operator node found")

    # [Additional operator handlers for ∨ and ¬ would follow similar pattern]

    def _split_args(self, expr: str) -> List[str]:
        """Splits logical expressions while handling nested parentheses"""
        parts = []
        depth = 0
        start = 0
        
        for i, c in enumerate(expr):
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            elif c == ',' and depth == 0:
                parts.append(expr[start:i])
                start = i + 1
        
        parts.append(expr[start:])
        return parts