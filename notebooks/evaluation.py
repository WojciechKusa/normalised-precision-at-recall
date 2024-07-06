import numpy as np


def average_precision_for_query(
    rankings: dict[str, float], qrels: dict[str, int]
) -> float:
    """Calculates the average precision for a given query.
    :param rankings: a dictionary of document ids and their corresponding scores
    :param qrels: a dictionary of document ids and their corresponding relevance labels
    :return: mean average precision for a given query
    """
    average_precision = 0
    tp, fp = 0, 0

    relevant_docs = sum(qrels.values())

    for rank, (doc_id, _) in enumerate(rankings.items()):
        if qrels.get(doc_id, 0) == 1:  # Document is relevant
            tp += 1
            precision = tp / (tp + fp)
            average_precision += precision
        else:
            fp += 1

    return average_precision / relevant_docs if relevant_docs > 0 else 0


def sqrt_n_precision_at_recall_for_query(
    rankings: dict[str, float], qrels: dict[str, int], recall_level: float = 0.95
) -> float:
    n_precision = n_precision_at_recall_for_query(rankings, qrels, recall_level)

    return np.sqrt(n_precision)


def n_precision_at_recall_for_query(
    rankings: dict[str, float], qrels: dict[str, int], recall_level: float = 0.95
) -> float:
    n_precision = None
    tp, fp, tn, fn = 0, 0, 0, 0

    relevant_docs = sum(qrels.values())
    total_docs = len(qrels)
    non_relevant_docs = total_docs - relevant_docs

    for rank, (doc_id, _) in enumerate(rankings.items()):
        if qrels.get(doc_id, 0) == 1:  # Document is relevant
            tp += 1
        else:
            fp += 1

        recall = tp / (tp + fn + relevant_docs - tp)

        if recall >= recall_level:
            tn = non_relevant_docs - fp
            n_precision = (tp * tn) / ((fp + tn) * (tp + fp))
            break

    return n_precision


def precision_at_recall_for_query(
    rankings: dict[str, float], qrels: dict[str, int], recall_level: float = 0.95
) -> float:
    precision = 0
    tp, fp, tn, fn = 0, 0, 0, 0

    relevant_docs = sum(qrels.values())

    for rank, (doc_id, _) in enumerate(rankings.items()):
        if qrels.get(doc_id, 0) == 1:  # Document is relevant
            tp += 1
        else:
            fp += 1

        recall = tp / (tp + fn + relevant_docs - tp)

        if recall >= recall_level:
            precision = tp / (tp + fp)
            break

    return precision


def tnr_at_recall_for_query(
    rankings: dict[str, float], qrels: dict[str, int], recall_level: float = 0.95
) -> float:
    tnr = 0
    tp, fp, tn, fn = 0, 0, 0, 0

    relevant_docs = sum(qrels.values())
    total_docs = len(qrels)
    non_relevant_docs = total_docs - relevant_docs

    for rank, (doc_id, _) in enumerate(rankings.items()):
        if qrels.get(doc_id, 0) == 1:  # Document is relevant
            tp += 1
        else:
            fp += 1

        recall = tp / (tp + fn + relevant_docs - tp)

        if recall >= recall_level:
            tn = non_relevant_docs - fp
            fn = relevant_docs - tp
            tnr = tn / (tn + fp)
            break

    return tnr


def find_last_relevant_for_query(
    rankings: dict[str, float], qrels: dict[str, int]
) -> float:
    relevant_docs = [doc_id for doc_id, rel in qrels.items() if rel > 0]

    docs = rankings

    # Find the position of the last relevant document
    last_relevant_position = None
    for doc_id, _ in reversed(docs.items()):
        if doc_id in relevant_docs:
            last_relevant_position = (
                list(docs.keys()).index(doc_id) + 1
            )  # Adding 1 as indexing starts from 0
            break

    # If a relevant document is found in the run
    if last_relevant_position is not None:
        lr = last_relevant_position / len(docs) * 100

    if last_relevant_position is None:
        lr = 100.0

    return lr
