'''
Description: Convert Onnx Model to Tree and Search Ops Fusion
'''
import onnx
from onnx.tools import update_model_dims
import numpy as np
import onnx.helper as helper
from onnx import shape_inference, TensorProto
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("[ONNX2TREE]")

ONNX_DTYPE = {
    0: TensorProto.FLOAT,
    1: TensorProto.FLOAT,
    2: TensorProto.UINT8,
    3: TensorProto.INT8,
    4: TensorProto.UINT16,
    5: TensorProto.INT16,
    6: TensorProto.INT32,
    7: TensorProto.INT64,
    8: TensorProto.STRING,
    9: TensorProto.BOOL
}

class OpPatternKind:
    # Pattern categories for operation types
    kElemWise = 0,                # Element-wise operation
    kBroadcast = 1,               # Broadcasting operator
    kInjective = 2,               # Injective operator
    kCommReduce = 3,              # Communicative reduction operator
    kOutEWiseFusable = 4,         # Fusable complex operations with elemwise
    kTuple = 7,                   # Tuple nodes, fusable to injective ops
    kOpaque = 8                   # Opaque operation, not fusable

ONNX_OPPATTERN = {
    "Conv": OpPatternKind.kOutEWiseFusable,
    "MaxPool": OpPatternKind.kOutEWiseFusable,
    'GlobalAveragePool': OpPatternKind.kOutEWiseFusable,
    "Relu": OpPatternKind.kElemWise,
    "BatchNormalization": OpPatternKind.kBroadcast,
    "Add": OpPatternKind.kBroadcast,
    "sqrt": OpPatternKind.kElemWise,
    "divide": OpPatternKind.kBroadcast,
    "Sqrt": OpPatternKind.kBroadcast,
    "Mul": OpPatternKind.kBroadcast,
    "expand_dims": OpPatternKind.kBroadcast,
    "negative": OpPatternKind.kElemWise,
    "Constant": OpPatternKind.kOpaque,
    "Gemm": OpPatternKind.kBroadcast,
    "Reshape": OpPatternKind.kOpaque,
}

NPDTYPE_2_ONNXDTYPE = {
    "float64": TensorProto.FLOAT,
    "float32": TensorProto.FLOAT,
    "uint8": TensorProto.UINT8,
    "int8": TensorProto.INT8,
    "uint16": TensorProto.UINT16,
    "int16": TensorProto.INT16,
    "int32": TensorProto.INT32,
    "int64": TensorProto.INT64,
    "str": TensorProto.STRING,
    "boolean": TensorProto.BOOL
}
# Load ONNX model and set initial nodes
# This section loads an ONNX model file and initializes its graph, 
# separating nodes into constant initializers and graph computation nodes.
onnx_model = onnx.load("./resnet50_v2_7.onnx")  # Load the specified ONNX model
graph = onnx_model.graph  # Extract the computational graph from the model
onnx_constant_nodes = graph.initializer  # Get constant nodes used as initial parameters (weights, biases, etc.)
onnx_nodes = graph.node  # Get graph nodes representing individual operations in the model

class GraphNode:
    """
    Represents a computational node in the graph. Tracks its name, outputs, 
    index, pattern, reference, and external reference status. Acts as a 
    wrapper around ONNX nodes to manage additional metadata.
    """
    def __init__(self):
        self.name = None  # Node name, unique identifier for each node
        self.outputs = []  # List of outgoing edges or links to other nodes
        self.index = 0  # Index of this node within the graph (for ordering)
        self.ref = None  # Reference to the actual ONNX node or initializer
        self.extern_ref = 0  # Marks if the node is an external reference (input/output)
        self.pattern = OpPatternKind.kOpaque  # Operation pattern, helps in fusing nodes

class LinkNode:
    """
    Represents a link or edge between two nodes in the graph, containing 
    pattern information to describe the type of operation in the link.
    """
    def __init__(self):
        self.value = None  # The source node this link originates from
        self.pattern = 0  # Pattern associated with this link (operation type)
        self.next = None  # Pointer to the next link in case of multiple edges

class Group:
    """
    Represents a collection or group of nodes for managing the hierarchical 
    structure within the graph. Used to find the root node and organize nodes 
    into clusters for efficient processing.
    """
    def __init__(self):
        self.parent = None  # Parent group node, representing hierarchy
        self.pattern = 0  # Pattern representing group operation type
        self.root_ref = None  # Root node of this group
        self.master_ref = None  # Main node reference for fusing within the group
        self.name = None  # Name or identifier of the group
        self.num_nodes = 1  # Number of nodes within this group

    def FindRoot(self):
        """
        Finds the root of the group by following parent references. Adjusts 
        all nodes' parent pointers within the group to point directly to the root.
        """
        if self.parent is None:
            return self
        else:
            root = self
            while(root.parent is not None):  # Follow parent links until reaching the root
                root = root.parent
            # Update all nodes in the path to point directly to the root
            while(self != root):
                parent = self.parent
                self.parent = root
                self = parent
        return root

class Graph:
    """
    Manages the graph structure, including nodes and edges. Allows adding 
    nodes, updating connections, and constructing a directed acyclic graph (DAG).
    """
    def __init__(self):
        self.edge_node_dict = {}  # Dictionary to map node names to their GraphNode instances
        self.post_dfs_order = []  # List to store nodes in post-order traversal order
        self.visited_list = []  # List to track visited nodes for DFS
        self.added_dict = {}  # Dictionary to map node names to indices in post_dfs_order
        self.root_flag = 1  # Flag to track if root node is set
        self.root_flag_1 = 1  # Secondary flag for managing root node setting

    def FindNode(self, node_name, nodes):
        """
        Searches for a node by its name in the list of computation nodes and constants.
        Returns the node and its type (either 'node' for computation node or 'var' for constant).
        """
        for node in nodes:  # Search computation nodes first
            if node_name in node.output:
                return node, "node"
        for init in onnx_constant_nodes:  # Search constant initializers if not found in computation nodes
            if node_name == init.name:
                return init, "var"
        logger.info("cannot find node {0}".format(node_name))  # Log if node is not found
        return None, None  # Return None if node is not found

    def Update(self, node, parent, pattern):
        """
        Updates or creates a new GraphNode with the specified pattern and parent link.
        Adds the link as an outgoing edge for the parent node.
        """
        # Check if node is already in the edge_node_dict, if so, get the current GraphNode
        if node.name in self.edge_node_dict.keys():
            current = self.edge_node_dict[node.name]
        else:
            current = GraphNode()  # Create new GraphNode if not found
        # For computation nodes, link with parent node if provided
        if node in onnx_nodes:
            if parent is not None:  # Create a LinkNode to connect with parent
                link = LinkNode()
                if parent.name not in self.edge_node_dict.keys():
                    logger.error("cannot find node {0} in edge dict, prob this is the last node".format(parent.name))
                    exit(1)
                parent = self.edge_node_dict[parent.name]  # Retrieve parent GraphNode
                link.value = parent  # Set the source node for the link
                link.pattern = pattern  # Set pattern type for link
                current.name = node.name
                current.outputs.append(link)  # Add link to the current node’s outputs
            else:
                current.name = node.name  # Set name for nodes without a parent (usually root nodes)
                current.extern_ref = 1  # Mark as external reference if no parent is linked
        return current

    def AddNode(self, node, node_pattern):
        """
        Adds a node to the graph and updates its index and pattern. Tracks node's 
        position in post-order for future processing.
        """
        if node.name not in self.edge_node_dict.keys():  # Ensure node is in edge_node_dict
            logger.error("cannot find node {0} in edge dict, prob this is the last node".format(node.name))
            exit(1)
        current = self.edge_node_dict[node.name]  # Retrieve the GraphNode instance
        current.index = len(self.post_dfs_order)  # Assign post-order index
        current.ref = node  # Set reference to the actual node
        current.pattern = node_pattern  # Set node's operation pattern
        logger.info("[add node] {0} {1} ".format(current.index, node.name))
        # Track node in added_dict to avoid duplication
        if node.name not in self.added_dict.keys():
            self.post_dfs_order.append(current)  # Append node to post-order traversal list
            self.added_dict[node.name] = current.index
        else:  # Update existing node in post-order list if it was added earlier
            index = self.added_dict[node.name]
            self.post_dfs_order[index] = current

    def VisitExpr(self, node):
        """
        Recursively builds the DAG for the model starting from the root node 
        and traversing inputs. Uses DFS to ensure proper traversal order.
        """
        if node == None or node in self.visited_list:  # Skip null or already visited nodes
            return 
        if self.root_flag:  # If this is the root node, initialize the edge_node_dict with it
            edge_root_node = self.Update(node, None, OpPatternKind.kOpaque)
            self.edge_node_dict[node.name] = edge_root_node
            self.root_flag = 0  # Disable the root_flag after root initialization
        op_pattern = ONNX_OPPATTERN[node.op_type]  # Get the operation pattern for this node type
        
        for input_s in node.input:  # Traverse each input of the node
            edge_pattern = op_pattern
            if input_s == "Placeholder_orig":  # Skip if input is a placeholder
                break
            input_node, node_type = self.FindNode(input_s, onnx_nodes)  # Find the input node by name
            if node_type == "node":  # If input is a computation node
                edge_node = self.Update(input_node, node, edge_pattern)  # Update the graph with the input node and its edge
                self.edge_node_dict[input_node.name] = edge_node
                self.VisitExpr(input_node)  # Recursively visit this input node
                self.visited_list.append(input_node)  # Mark input node as visited
            elif node_type == "var":  # If input is a constant variable
                self.visited_list.append(input_node)  # Mark variable as visited without recursion
        self.AddNode(node, op_pattern)  # Add the node to the graph after processing all inputs
        return
class DominatorTree:
    """
    Constructs a dominator tree for a graph, organizing nodes into groups based 
    on dominance relationships, which helps in efficient fusion of operations 
    later in the process.
    """
    def __init__(self):
        super().__init__()
        self.groups = []  # List of groups representing clusters of nodes in the graph
        self.tree_nodes = []  # List of TreeNode instances representing dominator tree nodes
        
    class TreeNode:
        """
        Represents a node in the dominator tree. Tracks the name, depth, pattern,
        parent in the dominator tree, and reference to the corresponding graph node.
        """
        def __init__(self):
            self.name = None  # Name of the node (unique identifier)
            self.parent = None  # Parent node in the dominator tree
            self.depth = 0  # Depth of this node within the tree (root node has depth 1)
            self.pattern = None  # Operation pattern, to identify fusion possibilities
            self.index = 0  # Index in the post-order traversal of the graph
            self.gnode = None  # Reference to the associated graph node
    
    def InitGropus(self, graph):
        """
        Initializes groups for each node in the graph. Each group represents 
        a cluster with a specific pattern type and root reference.
        """
        size = len(graph.post_dfs_order)
        for index in range(size):
            graph_node = graph.post_dfs_order[index]  # Get node in post-order
            group_node = Group()  # Create a new group for the node
            group_node.pattern = graph_node.pattern  # Assign pattern type
            group_node.root_ref = graph_node.ref  # Set root reference to the graph node
            group_node.name = graph_node.name  # Set the group name as the node name
            if group_node.pattern == OpPatternKind.kOutEWiseFusable:  # Mark as master if fusable
                group_node.master_ref = graph_node.ref
            self.groups.append(group_node)  # Add the group to the groups list

    def CombinePattern(self, lhs, rhs):
        """
        Combines two operation patterns to determine the appropriate edge 
        pattern for a fusion operation, based on priority.
        """
        return lhs if lhs > rhs else rhs  # Return the higher priority pattern

    def LeastCommonAncestorMulEdges(self, lhs, rhs, edge_pattern):
        """
        Finds the least common ancestor (LCA) between two nodes in scenarios 
        with multiple edges. Adjusts the edge pattern based on the encountered nodes.
        """
        while lhs != rhs:  # Continue until both nodes reach the same ancestor
            if lhs is None or rhs is None:
                return None
            if lhs.depth < rhs.depth:  # Move the deeper node up the tree
                edge_pattern = self.CombinePattern(edge_pattern, rhs.pattern)
                rhs = rhs.parent
            elif rhs.depth < lhs.depth:
                edge_pattern = self.CombinePattern(edge_pattern, lhs.pattern)
                lhs = lhs.parent
            else:  # Move both nodes up simultaneously if at same depth
                edge_pattern = self.CombinePattern(edge_pattern, lhs.pattern)
                edge_pattern = self.CombinePattern(edge_pattern, rhs.pattern)
                lhs = lhs.parent
                rhs = rhs.parent
        return lhs  # Return the LCA node
    
    def LeastCommonAncestor(self, edges, edge_pattern, index):
        """
        Finds the least common ancestor for a list of node edges, which represents 
        the first node that dominates all others. Used for fusion eligibility checks.
        """
        if len(edges) <= index:
            return None
        link_head = edges[index]  # Get the first edge to start comparison
        def get_node(father_node):
            oindex = father_node.index
            return self.tree_nodes[oindex]  # Retrieve the corresponding tree node
        parent = get_node(link_head.value)  # Initial parent is the first edge’s node
        edge_pattern = link_head.value.pattern  # Set initial edge pattern
        index += 1  # Increment index to check next edge
        for i in range(index, len(edges)):  # Iterate through remaining edges
            link = edges[index]
            parent = self.LeastCommonAncestorMulEdges(parent, get_node(link.value), edge_pattern)
            edge_pattern = self.CombinePattern(edge_pattern, link.value.pattern)  # Adjust pattern
        return parent  # Return the least common ancestor of all edges
        
    def GetNode(self, graph_node, graph):
        """
        Creates a tree node for the dominator tree from a given graph node, setting 
        depth, pattern, and parent attributes based on least common ancestor.
        """
        tree_node = self.TreeNode()
        if graph_node.extern_ref == 1:  # If the node is an external reference
            tree_node.name = graph_node.name
            tree_node.depth = 1  # Root nodes start at depth 1
            tree_node.parent = None
            tree_node.pattern = "kOpaque"  # Non-fusable node pattern
            tree_node.parent_gnode = graph_node
        else:  # If the node has an internal reference
            pattern = OpPatternKind.kElemWise
            tree_node.name = graph_node.name
            parent = self.LeastCommonAncestor(graph_node.outputs, pattern, 0)  # Get parent as LCA
            tree_node.depth = parent.depth + 1 if parent else 1  # Set depth relative to parent
            tree_node.parent = parent  # Assign parent node
            tree_node.pattern = pattern  # Assign pattern type
            parent_gnode = None
            for node in graph:
                if node.name == parent.name:
                    parent_gnode = node  # Find the corresponding parent node in graph
            assert parent_gnode is not None
            tree_node.parent_gnode = parent_gnode
            logger.info("[dom node] {0} {1} {2}".format(tree_node.depth, graph_node.name, tree_node.parent_gnode.name))
        return tree_node  # Return the constructed tree node
    
    def PostDom(self, graph):
        """
        Constructs dominator tree nodes for all graph nodes in post-order. 
        This allows for efficient fusion by building the tree from root to leaves.
        """
        size = len(graph.post_dfs_order)
        self.tree_nodes = [None] * size
        for i in range(size, 0, -1):  # Traverse nodes in reverse post-order
            self.tree_nodes[i-1] = self.GetNode(graph.post_dfs_order[i-1], graph.post_dfs_order)

    def DominatorPartition(self, graph):
        """
        Partitions the graph into groups using the dominator tree structure, 
        creating the tree and initializing groups for each node.
        """
        self.InitGropus(graph)  # Initialize groups for each node in graph
        self.PostDom(graph)  # Build dominator tree nodes


class FuseOps:
    """
    Manages and executes fusion operations on nodes within the dominator tree. 
    Ensures eligible nodes are combined to optimize graph structure.
    """
    def __init__(self):
        self.fuse = None  # Placeholder for fusion information
        self.visited = []  # List to track visited nodes during path checks
        
    def CheckPath_(self, src, sink, fcond, tree):
        """
        Recursively checks if a path exists from the source node to the sink 
        node that satisfies the fusion condition. Verifies fusion eligibility.
        """
        if src.name in self.visited:  # Avoid revisiting nodes
            return True
        self.visited.append(src.name)
        gnode = tree.groups[src.index]
        assert gnode is not None
        gnode = gnode.FindRoot()  # Get root of the current group
        if not fcond(gnode.pattern, src == sink):  # Check fusion condition
            return False
        if src == sink:  # If reached sink, return true
            return True
        for link in src.outputs:  # Recursively check all outgoing edges
            if not self.CheckPath_(link.value, sink, fcond, tree):
                return False
        return True
        
    def CheckPath(self, src, sink, fcond, tree):
        """
        Initializes path checking from source to sink nodes to determine if 
        fusion conditions are met. Clears visited nodes at the start.
        """
        assert src.extern_ref == 0, "Root node error"  # Ensure src is not an external reference
        self.visited = []  # Reset visited nodes for fresh path check
        assert src != sink  # Source and sink should be different nodes
        for link in src.outputs:  # Begin path check from each output of src
            if not self.CheckPath_(link.value, sink, fcond, tree):
                return False
        return True
    
    def MergeFromTo(self, child, parent):
        """
        Merges one group (child) into another (parent), combining their nodes 
        and patterns. Updates parent references accordingly.
        """
        child = child.FindRoot()
        parent = parent.FindRoot()
        if child == parent:  # If already same group, do nothing
            return
        parent.num_nodes += child.num_nodes  # Increment parent node count
        child.parent = parent  # Set parent as new root for child
        if child.master_ref is not None:  # Update master reference if child has it
            assert parent.master_ref is None
            parent.master_ref = child.master_ref
            parent.pattern = child.pattern
        else:  # Else retain parent's master reference and pattern
            assert parent.master_ref is not None
            child.master_ref = parent.master_ref
            child.pattern = parent.pattern
        
    def CommitFuse_(self, src, sink, target, tree):
        """
        Executes the fusion operation from source to sink along a path, 
        merging groups and updating their relationships.
        """
        if src == sink or src.name in self.visited:
            return
        self.visited.append(src.name)
        gnode = tree.groups[src.index]
        assert gnode is not None
        self.MergeFromTo(gnode, target)  # Merge source group into target
        for link in src.outputs:  # Recursively apply fusion along path
            self.CommitFuse_(link.value, sink, target, tree)
            
    def CommitFuse(self, src, sink, tree):
        """
        Wrapper for committing fusion operations between two nodes. Initializes
        the target group and starts the fusion process.
        """
        target = tree.groups[sink.index]  # Target group is sink’s group
        logger.info("[Merge] {0} + {1} -> {2}".format(src.name, sink.name, target.name))
        self.visited = []
        assert src != sink
        self.CommitFuse_(src, sink, target, tree)  # Start recursive fusion

    def RunFuse(self, graph, tree):
        """
        Executes the fusion algorithm in multiple phases. Each phase attempts to 
        fuse nodes based on certain conditions to optimize the graph.
        """
        def fcond0(kind, issink):
            # Defines a fusion condition where broadcasting operators are fusable
            if isinstance(kind, int):
                kind = (kind,)
            return kind <= OpPatternKind.kBroadcast  # Fusion condition for broadcast
        for phase in range(0, 1):  # Loop over fusion phases (currently only 1 phase)
            for i in range(0, len(tree.groups)):  # Iterate through groups in tree
                graph_node = graph.post_dfs_order[i]
                dom_node = tree.tree_nodes[i]
                group_node = tree.groups[i]
                if dom_node is not None and group_node.pattern == OpPatternKind.kOutEWiseFusable:
                    if phase != 0:  # Skip further phases if not first
                        continue
                    if dom_node.parent is not None and dom_node.pattern == OpPatternKind.kElemWise:
                        logger.info("[fuse node] {0} {1}".format(group_node.name, dom_node.parent.name))
                        if self.CheckPath(graph_node, dom_node.parent_gnode, fcond0, tree):  # Check fusion eligibility
                            self.CommitFuse(graph_node, dom_node.parent_gnode, tree)  # Perform fusion
            for node in tree.groups:  # Log final groups with master references
                if node.master_ref is not None:
                    logger.info("[groups] {0} {1} {2}".format(node.name, node.num_nodes, node.master_ref.name))

if __name__ == "__main__":
    # Main execution to build and optimize the graph with fusion
    topo_graph = Graph()
    topo_graph.VisitExpr(onnx_nodes[-1])  # Start building graph from last ONNX node
    
    post_dom_tree = DominatorTree()
    post_dom_tree.DominatorPartition(topo_graph)  # Create dominator tree
    
    fuse_op_object = FuseOps()
    fuse_op_object.RunFuse(topo_graph, post_dom_tree)  # Execute fusion on the dominator tree