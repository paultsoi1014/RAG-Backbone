from typing import List, Optional, Tuple
from llama_index.core.schema import NodeWithScore


def repack_document(
    nodes: List[NodeWithScore], strategy: Optional[str] = "sides"
) -> Tuple[List[NodeWithScore], str]:
    """
    Repack the nodes and text based on the specified strategy

    Parameters
    ----------
    nodes: List[NodeWithScore]
        List of nodes ordered by relevancy score (highest to lowest)
    strategy: Optional[str]
        The repacking strategy: 'forward', 'reverse', or 'sides', defaults to
        'sides'

    Returns
    -------
    Tuple[List[NodeWithScore], str]
        - Repacked nodes according to the strategy
        - Concatenated string of repacked text content
    """
    if strategy not in ["forward", "reverse", "sides"]:
        raise ValueError("strategy must be one of: 'forward', 'reverse', 'sides'")

    # Extract text content from nodes
    text_list = [node.node.get_content() for node in nodes]

    if strategy == "forward":
        # Forward: Keep original order (most relevant first)
        repacked_nodes = nodes.copy()
        repacked_text = text_list.copy()

    elif strategy == "reverse":
        # Reverse: Reverse the order (least relevant first, most relevant last)
        repacked_nodes = nodes[::-1]
        repacked_text = text_list[::-1]

    elif strategy == "sides":
        # Sides: Most relevant at head and tail, less relevant in middle
        repacked_nodes = [None] * len(nodes)
        repacked_text = [None] * len(nodes)
        left, right = 0, len(nodes) - 1

        for i, (node, text) in enumerate(zip(nodes, text_list)):
            if i % 2 == 0:
                # Even index: place at the beginning (head)
                repacked_nodes[left] = node
                repacked_text[left] = text
                left += 1
            else:
                # Odd index: place at the end (tail)
                repacked_nodes[right] = node
                repacked_text[right] = text
                right -= 1

    return repacked_nodes, "\n\n".join(repacked_text)
