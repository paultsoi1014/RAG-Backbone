from collections import defaultdict
from llama_index.core.schema import NodeWithScore


def calculate_relevancy(nodes: list[NodeWithScore]) -> dict:
    """
    Calculate document relevancy scores based on retrieved nodes using
    the CombMNZ method and normalize them into percentages

    Parameters
    ----------
    nodes: list[NodeWithScore]
        List of retrieved nodes with their associated scores

    Returns
    -------
    dict
        A dictionary mapping document names to their relevancy scores as
        percentages
    """
    # group score by document name
    doc_chunks = defaultdict(list)

    for node in nodes:
        collection = node.metadata.get("collection", "unknown_collection")
        doc_chunks[collection].append(node.score)

    # calculated weighted score using CombMNZ
    raw_scores = {}
    for doc, vals in doc_chunks.items():
        score_sum = sum(vals)
        count = len(vals)
        raw_scores[doc] = score_sum * count

    # Normalize score into percentages
    total_score = sum(raw_scores.values())
    percentages = {
        doc: (round(val / total_score * 100, 2)) if total_score > 0 else 0
        for doc, val in raw_scores.items()
    }

    return percentages
