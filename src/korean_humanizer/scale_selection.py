"""Exact genre quotas with a shared per-author cap, solved as an integral flow."""

from collections import Counter, defaultdict, deque


def select_exact(candidates, quotas, *, author_cap=20):
    """Return source IDs or None if the exact quota is infeasible. Never pad or duplicate."""
    ids = [row["case_id"] for row in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate source IDs in selection pool")
    grouped = defaultdict(list)
    for row in sorted(candidates, key=lambda r: r["case_id"]):
        if row["genre"] in quotas:
            grouped[(row["author_group"], row["genre"])].append(row["case_id"])
    source = ("root",)
    sink = ("sink",)
    graph = defaultdict(dict)

    def edge(a, b, capacity):
        graph[a][b] = capacity
        graph[b][a] = 0

    for author in sorted({a for a, g in grouped}):
        edge(source, ("author", author), author_cap)
    for (author, genre), rows in sorted(grouped.items()):
        edge(("author", author), ("genre", genre), len(rows))
    for genre, count in sorted(quotas.items()):
        if count < 0:
            raise ValueError("negative genre quota")
        edge(("genre", genre), sink, count)
    total = 0
    while True:
        previous = {source: None}
        queue = deque([source])
        while queue and sink not in previous:
            node = queue.popleft()
            for dest, cap in graph[node].items():
                if cap > 0 and dest not in previous:
                    previous[dest] = node
                    queue.append(dest)
        if sink not in previous:
            break
        increment = sum(quotas.values()) - total
        node = sink
        while previous[node] is not None:
            increment = min(increment, graph[previous[node]][node])
            node = previous[node]
        node = sink
        while previous[node] is not None:
            parent = previous[node]
            graph[parent][node] -= increment
            graph[node][parent] += increment
            node = parent
        total += increment
        if total == sum(quotas.values()):
            break
    if total != sum(quotas.values()):
        return None
    selected = []
    for (author, genre), rows in sorted(grouped.items()):
        used = graph[("genre", genre)][("author", author)]
        selected.extend(rows[:used])
    lookup = {row["case_id"]: row for row in candidates}
    assert Counter(lookup[key]["genre"] for key in selected) == {
        g: n for g, n in quotas.items() if n
    }
    assert (
        max(Counter(lookup[key]["author_group"] for key in selected).values(), default=0)
        <= author_cap
    )
    return selected
