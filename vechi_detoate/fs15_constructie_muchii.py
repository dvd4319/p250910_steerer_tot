import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def build_edge_to_triangles(tris):
    """
    Construiește dict: edge(tuple(sorted(a,b))) -> list(triangle_index)
    tris: (n_tris,3) array
    """
    edge2tris = defaultdict(list)
    for t_idx, tri in enumerate(tris):
        for i, j in ((0,1),(1,2),(2,0)):
            a, b = int(tri[i]), int(tri[j])
            edge = tuple(sorted((a, b)))
            edge2tris[edge].append(t_idx)
    return edge2tris

def find_interface_edges(edge2tris, tri_domains):
    """
    Returnează lista de muchii care separă două triunghiuri cu tag-uri diferite.
    - edge2tris: dict edge -> [tri_idx, ...]
    - tri_domains: array cu tag-ul domeniului pentru fiecare triunghi (lungime == tris.shape[0])
    """
    interface_edges = []
    interface_edge_domains = []  # pereche (d1,d2) corespunzătoare muchiei
    for edge, tris_adj in edge2tris.items():
        if len(tris_adj) == 2:
            t0, t1 = tris_adj
            d0 = int(tri_domains[t0]) if t0 < len(tri_domains) else 0
            d1 = int(tri_domains[t1]) if t1 < len(tri_domains) else 0
            if d0 != d1:
                interface_edges.append(edge)
                interface_edge_domains.append((d0, d1))
        # daca len(tris_adj)==1 -> muchie la exterior; nu o considerăm "separatoare de domenii"
    return interface_edges, interface_edge_domains

def group_edges_by_domain(interface_edges, interface_edge_domains):
    """
    Din lista de muchii si perechile de domenii returneaza:
    - edges_by_domain: dict dom -> set of edges (tuples)
    - nodes_by_domain: dict dom -> set of node indices care apar pe muchiile de interfata ale domeniului
    """
    edges_by_domain = defaultdict(set)
    nodes_by_domain = defaultdict(set)
    for edge, (d0, d1) in zip(interface_edges, interface_edge_domains):
        a, b = edge
        # adaugam muchia pentru ambele domenii implicate (asta e util pentru desen per domeniu)
        edges_by_domain[d0].add(edge)
        edges_by_domain[d1].add(edge)
        nodes_by_domain[d0].update([a, b])
        nodes_by_domain[d1].update([a, b])
    # convertește la liste sortate
    edges_by_domain = {dom: sorted(list(es)) for dom, es in edges_by_domain.items()}
    nodes_by_domain = {dom: sorted(list(ns)) for dom, ns in nodes_by_domain.items()}
    return edges_by_domain, nodes_by_domain

def plot_interface_edges(nodes, edges_by_domain, nodes_by_domain, title="Domain interface edges"):
    """
    Desenează muchiile de interfata colorate per domeniu și marchează nodurile (opțional).
    """
    n_domains = max(edges_by_domain.keys()) + 1 if edges_by_domain else 1
    cmap = plt.cm.get_cmap("tab20", max(1, len(edges_by_domain)))

    plt.figure(figsize=(10,10))
    # optional: desenam mesh slab (fara muchii) pentru context
    # plt.triplot(nodes[:,0], nodes[:,1], tris, color='0.8', linewidth=0.3)

    for i, dom in enumerate(sorted(edges_by_domain.keys())):
        color = cmap(i)
        edges = edges_by_domain[dom]
        # desenare muchii
        for (a,b) in edges:
            xa, ya = nodes[a,0], nodes[a,1]
            xb, yb = nodes[b,0], nodes[b,1]
            plt.plot([xa, xb], [ya, yb], color=color, linewidth=1.6)
        # desenare noduri de interfata (mai mici)
        nds = np.array(nodes_by_domain.get(dom, []), dtype=int)
        if nds.size:
            plt.scatter(nodes[nds,0], nodes[nds,1], s=6, marker='o', color=color, label=f"Domain {dom}")

    plt.gca().set_aspect("equal")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc='best', fontsize='small', ncol=2)
    plt.show()

def save_domain_node_lists(nodes_by_domain, prefix="domain_boundary_nodes"):
    """
    Salvează listele nodurilor per domeniu în fișiere text: prefix_dom<N>.txt
    """
    for dom, nlist in nodes_by_domain.items():
        fname = f"{prefix}_dom{dom}.txt"
        np.savetxt(fname, np.array(nlist, dtype=int), fmt="%d")
        print(f"Saved {len(nlist)} nodes for domain {dom} -> {fname}")

