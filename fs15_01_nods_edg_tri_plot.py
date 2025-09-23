'''
Scriptul contine 3 functii: 
---------------------------------
0. clean_line(line: str)
1. load_comsol_mphtxt(filename)
2. build_adjacency(tris, edgs)
3. find_domains(tris, adjacency)
----------------------------------
'''
import numpy as np
from collections import defaultdict, deque
import matplotlib.pyplot as plt
from mpi4py import MPI
import matplotlib.colors as mcolors
import numpy as np
import meshio
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from matplotlib.patches import Polygon as Polygon1 
from shapely.ops import unary_union
import matplotlib.tri as mtri

###############################################
from dolfinx import io, fem
from dolfinx.io import XDMFFile
from dolfinx import io, plot
import pyvista as pv
import pyvista
###############################################
from m_triangle_manual.m_edgs_domains import edgs_domains_m
from m_triangle_manual.m_edgs_elements import edges_elements_m
from m_triangle_manual.m_edgs_params import edges_params_m
from m_triangle_manual.m_edgs_up_down import edgs_up_down_m
### ======================================= ### 
from m_triangle_manual.m_mesh_point_coordinates import mesh_point_coordinates_m 
### ======================================= ### 
from m_triangle_manual.m_tri_domains import tri_domains_m
from m_triangle_manual.m_tri_elements import tri_elements_m
### ======================================= ### 
from m_triangle_manual.m_vtx_domains import vtx_domains_m
from m_triangle_manual.m_vtx_elements import vtx_elements_m

# from matplotlib.collections import Patch
###############################################
##################################

############################################################################
################ (0) FUNCITA ZERO ########################################
############################################################################
def clean_line(line: str):
    """Elimină comentariile după # și întoarce lista de token-uri"""
    return line.split("#")[0].strip().split()


############################################################################
################ (1) PRIMA FUNCITE ########################################
############################################################################

def f01_load_comsol_mphtxt(filename):
    nodes = []
    triangles = []
    edges = []
    tri_domains = []
    tri_domains1 = []
    with open(filename) as f:
        lines = [line.strip() for line in f if line.strip()]

    i = 0
    while i < len(lines):
        line = lines[i]

        # --- Secțiunea noduri ---
        if line.startswith("# Mesh point coordinates"):
            i += 1
            while i < len(lines) and not lines[i].startswith("#"):
                parts = clean_line(lines[i])
                if len(parts) >= 2:
                    try:
                        x, y = map(float, parts[:2])
                        nodes.append((x, y))
                    except ValueError:
                        pass
                i += 1
            continue

        # --- Elemente edg (muchii) ---
        if line.startswith("3 edg"):
            nnodes = int(clean_line(lines[i+1])[0])
            count = int(clean_line(lines[i+2])[0])
            i += 3
            k = 0
            while k < count and i + k < len(lines):
                parts = clean_line(lines[i+k])
                if len(parts) >= nnodes:
                    try:
                        edges.append(list(map(int, parts[:nnodes])))
                    except ValueError:
                        pass
                k += 1
            i += k
            continue

        # --- Elemente tri ---
        if line.startswith("3 tri"):
            nnodes = int(clean_line(lines[i+1])[0])
            count = int(clean_line(lines[i+2])[0])
            i += 3
            k = 0
            while k < count and i + k < len(lines):
                parts = clean_line(lines[i+k])
                if len(parts) >= nnodes:
                    try:
                        triangles.append(list(map(int, parts[:nnodes])))
                    except ValueError:
                        pass
                k += 1
            i += k
            # --- Citim domeniile triunghiurilor pentru toate triunghiurile ---
            # Cautăm secțiunea cu domeniile
            while i < len(lines):
                parts = clean_line(lines[i])
                if parts and all(p.isdigit() for p in parts):
                    tri_domains.extend([int(p) for p in parts])
                elif parts and parts[0].startswith("#"):  # am ieșit din secțiunea de domenii
                    break
                i += 1
            continue

        i += 1
    ######################################################
    with open("nodes.py", "w") as f:
        f.write("nodes = [\n")
        for x, y in nodes:
            f.write(f"    [{x}, {y}],\n")
        f.write("]\n")

    with open("edges.py", "w") as f:
        f.write("edges = [\n")
        for x, y in edges:
            f.write(f"    [{x}, {y}],\n")
        f.write("]\n")

    with open("triangles.py", "w") as f:
        f.write("triangles = [\n")
        for x, y, z in triangles:
            f.write(f"    [{x}, {y}, {z}],\n")
        f.write("]\n")

    with open("tri_domains.py", "w") as f:
        f.write("tri_domains = [\n")
        for x in tri_domains:
            f.write(f"    {x},\n")
        f.write("]\n")
    return np.array(nodes), np.array(triangles, dtype=int), np.array(edges, dtype=int), np.array(tri_domains, dtype=int)

#############################
###########################################################################
############### (2) A DOUA FUNCITE ########################################
###########################################################################

# funcție pentru corectarea tri_domains
def f04_convert_mphtxt_to_gmsh_nou(mphtxt_file,msh_file):
    nodes, tris, edgs, tri_domains = f01_load_comsol_mphtxt(mphtxt_file)
    tri_domains_correct = tri_domains[3:757]
    # print(f"tri domains INAINTE: {tri_domains}")
    # print(f"tri domains DUPA: {tri_domains_correct}")
    print(f"LUNGIME tri domains mshtxt INAINTE: {len(tri_domains)}")
    print(f"LUNGIME tri domains mshtxt DUPA: {len(tri_domains_correct)}")
    if len(tri_domains_correct) != len(tris):
        raise ValueError(f"lungimea tri_domains_correct ({len(tri_domains_correct)}) nu coincide cu lungimea tris ({len(tris)})")
    
    unique_domains = np.unique(tri_domains_correct)
    print(f"valori unice în tri_domains_correct (înainte de corecție): {unique_domains}")
    
    tri_domains_correct = np.where(tri_domains_correct == 0, 1, tri_domains_correct)
    
    unique_domains_corrected = np.unique(tri_domains_correct)
    if not np.all(np.isin(unique_domains_corrected, [1, 2, 3])):
        raise ValueError(f"tri_domains_correct conține valori neașteptate după corecție: {unique_domains_corrected}")
    
    with open(msh_file, "w") as out:
        out.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        out.write("$Nodes\n")
        out.write(f"{len(nodes)}\n")
        for i, (x, y) in enumerate(nodes, 1):
            out.write(f"{i} {x} {y} 0\n")
        out.write("$EndNodes\n")
        out.write("$Elements\n")
        out.write(f"{len(tris) + len(edgs)}\n")

        # vechi 
        # for i, (n1, n2) in enumerate(edgs, 1):
        #     out.write(f"{i} 1 2 0 0 {n1 + 1} {n2 + 1}\n")
        # for i, (t, dom) in enumerate(zip(tris, tri_domains_correct), len(edgs) + 1):
        #     out.write(f"{i} 2 2 {dom} {dom} {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")
        # noi 
        for i, (n1, n2) in enumerate(edgs, 1):
            out.write(f"{i} 1 2 {1} 0 {n1 + 1} {n2 + 1}\n")  # 1 ca physical, 0 ca geometric
        for i, (t, dom) in enumerate(zip(tris, tri_domains_correct), len(edgs) + 1):
            out.write(f"{i} 2 2 {dom} 0 {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")

        out.write("$EndElements\n")
    print(f"număr tri-domenii gmsh detectate: {len(unique_domains_corrected)}")
    return tri_domains_correct

#############################
###########################################################################
############### (2) A DOUA FUNCITE ########################################
###########################################################################

def f02_build_adjacency(tris, edgs):
    # Facem set de muchii de frontieră (cele roșii)
    boundary_edges = {tuple(sorted(e)) for e in edgs}

    # Construim graful de adiacență între triunghiuri
    adjacency = defaultdict(list)
    edge_to_tri = defaultdict(list)

    for t_idx, tri in enumerate(tris):
        for k in range(3):
            e = tuple(sorted((tri[k], tri[(k+1)%3])))
            edge_to_tri[e].append(t_idx)

    for e, tlist in edge_to_tri.items():
        if len(tlist) == 2 and e not in boundary_edges:
            # triunghiurile împart muchia și nu e frontieră
            t1, t2 = tlist
            adjacency[t1].append(t2)
            adjacency[t2].append(t1)

    return adjacency

###########################################################################
############### (3) A TREIA FUNCITE ########################################
###########################################################################
def f03_find_domains(tris, adjacency):
    visited = np.zeros(len(tris), dtype=bool)
    domains = np.full(len(tris), -1, dtype=int)
    domain_id = 0

    for t in range(len(tris)):
        if not visited[t]:
            # BFS/DFS pentru componenta conexă
            queue = deque([t])
            visited[t] = True
            domains[t] = domain_id
            while queue:
                cur = queue.popleft()
                for neigh in adjacency[cur]:
                    if not visited[neigh]:
                        visited[neigh] = True
                        domains[neigh] = domain_id
                        queue.append(neigh)
            domain_id += 1
    return domains


###########################################################################
############### (4) A PATRA FUNCȚIE #######################################
###########################################################################
import numpy as np

# #################################################################################
# def f04_convert_mphtxt_to_gmsh(mphtxt_file, msh_file):
#     """
#     Convert a COMSOL .mphtxt mesh to Gmsh .msh format.
#     Primele 3 valori din tri_domains sunt ignorate.
#     Triangles with invalid domains (<=0 or absurd) are remapate la 1.
#     """
#     # Load data from COMSOL mphtxt
#     # nodes, tris, edgs, tri_domains_raw = f01_load_comsol_mphtxt(mphtxt_file)
#     nodes = mesh_point_coordinates_m
#     edgs = edges_elements_m
#     tris = tri_elements_m
#     vtx = vtx_elements_m

#     tri_domains_raw =tri_domains_m
#     edgs_domains_raw = edgs_domains_m
#     vtx_domains_raw = vtx_domains_m
#     # --- Eliminăm primele 3 valori din tri_domains ---
#     # tri_domains_raw = tri_domains_raw[3:]

#     # --- Corectăm domeniile triunghiurilor ---
#     tri_domains = np.array([d if d > 0 else 1 for d in tri_domains_raw], dtype=int)
#     edgs_domains = np.array([d if d > 0 else 1 for d in edgs_domains_raw], dtype=int)
#     vtx_domains = np.array([d if d > 0 else 1 for d in vtx_domains_raw], dtype=int)

#     # tri_domains =  tri_domains_raw
#     # edgs_domains = edgs_domains_raw
#     # vtx_domains = vtx_domains_raw

#     # Safety check: asigură că avem tri_domains pentru fiecare triunghi
#     if len(tri_domains) < len(tris):
#         tri_domains = np.pad(tri_domains, (0, len(tris) - len(tri_domains)), constant_values=1)
#     elif len(tri_domains) > len(tris):
#         tri_domains = tri_domains[:len(tris)]

#     # --- Gmsh folosește 1-based indexing ---
#     tris = np.array(tris) + 1
#     edgs = np.array(edgs) + 1
#     vtx = np.array(vtx) + 1
#     # --- Scriere fișier Gmsh ---
#     with open(msh_file, "w") as out:
#         out.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")

#         # Nodes
#         out.write(f"$Nodes\n{len(nodes)}\n")
#         for i, (x, y) in enumerate(nodes, 1):
#             out.write(f"{i} {x} {y} 0.0\n")
#         out.write("$EndNodes\n")

#         # Elements
#         out.write(f"$Elements\n{len(tris) + len(edgs)}\n")
#         element_id = 1

#         # Triangles
#         for t, dom in zip(tris, tri_domains):
#             out.write(f"{element_id} 2 2 {dom} {dom} {t[0]} {t[1]} {t[2]}\n")
#             element_id += 1

#         # Edges
#         for e, dom in zip(edgs, edgs_domains):
#             out.write(f"{element_id} 1 2 0 0 {e[0]} {e[1]}\n")
#             element_id += 1

#         out.write("$EndElements\n")

#     print(f"Converted to Gmsh: {msh_file}")
#     print(f"Nodes written: {len(nodes)}")
#     print(f"Triangles written: {len(tris)}")
#     print(f"Edges written: {len(edgs)}")
#     print(f"Domains written: {len(set(tri_domains))} unique")

#     return tri_domains
#########################################################################################

# Funcția modificată
def f04_convert_mphtxt_to_gmsh(msh_file):
    """
    Converteste datele extrase din COMSOL .mphtxt în format Gmsh .msh.
    Ignoră primele 3 valori din tri_domains dacă este necesar, dar în datele tale pare că nu e cazul.
    Remapează domeniile invalide (<=0) la 1.
    """

    mesh_point_coordinates_m1 = np.array(mesh_point_coordinates_m)
    tri_elements_m1 = np.array(tri_elements_m)
    edges_elements_m1 = np.array(edges_elements_m)
    vtx_elements_m1 = np.array(vtx_elements_m)  # Nu e folosit în scriere, dar inclus pentru completitudine

    nodes = mesh_point_coordinates_m1
    tris = tri_elements_m1
    edgs = edges_elements_m1
    vtx = vtx_elements_m1  # Nu e folosit în scriere, dar inclus pentru completitudine

    tri_domains_m1 = np.array(tri_domains_m)
    edgs_domains_m1 = np.array(edgs_domains_m)
    vtx_domains_m1 = np.array(vtx_domains_m)

    tri_domains_raw = tri_domains_m1
    edgs_domains_raw = edgs_domains_m1
    vtx_domains_raw = vtx_domains_m1
    # Corectăm domeniile (remapăm <=0 la 1)
    tri_domains = np.array([d if d > 0 else 1 for d in tri_domains_raw], dtype=int)
    edgs_domains = np.array([d if d > 0 else 1 for d in edgs_domains_raw], dtype=int)
    vtx_domains = np.array([d if d > 0 else 1 for d in vtx_domains_raw], dtype=int)

    # Verifică lungimile
    if len(tri_domains) != len(tris):
        print(f"Avertisment: {len(tri_domains)} domenii triunghi vs {len(tris)} triunghiuri. Ajustez.")
        tri_domains = np.pad(tri_domains, (0, len(tris) - len(tri_domains)), constant_values=1) if len(tri_domains) < len(tris) else tri_domains[:len(tris)]

    if len(edgs_domains) != len(edgs):
        print(f"Avertisment: {len(edgs_domains)} domenii muchii vs {len(edgs)} muchii. Ajustez.")
        edgs_domains = np.pad(edgs_domains, (0, len(edgs) - len(edgs_domains)), constant_values=1) if len(edgs_domains) < len(edgs) else edgs_domains[:len(edgs)]

    # Gmsh folosește indexare 1-based
    tris = tris + 1
    edgs = edgs + 1
    vtx = vtx + 1  # Dacă e nevoie

    # Scriere fișier .msh
    with open(msh_file, "w") as out:
        out.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")

        # Nodes
        out.write(f"$Nodes\n{len(nodes)}\n")
        for i, (x, y) in enumerate(nodes, 1):
            out.write(f"{i} {x} {y} 0.0\n")
        out.write("$EndNodes\n")

        # Elements (triunghiuri + muchii; ignorăm vertex-urile dacă nu sunt necesare ca elemente separate)
        num_elements = len(tris) + len(edgs)
        out.write(f"$Elements\n{num_elements}\n")
        element_id = 1

        # Triunghiuri (tip 2 = triangle, 2 tags: physical group, geometrical entity)
        for t, dom in zip(tris, tri_domains):
            out.write(f"{element_id} 2 2 {dom} {dom} {t[0]} {t[1]} {t[2]}\n")
            element_id += 1

        # Muchii (tip 1 = line, 2 tags: 0 0 dacă nu sunt boundary specifice, sau folosește dom)
        for e, dom in zip(edgs, edgs_domains):
            out.write(f"{element_id} 1 2 {dom} {dom} {e[0]} {e[1]}\n")
            element_id += 1

        out.write("$EndElements\n")

    print(f"Converted to Gmsh: {msh_file}")
    print(f"Nodes: {len(nodes)}")
    print(f"Triangles: {len(tris)}")
    print(f"Edges: {len(edgs)}")
    print(f"Unique triangle domains: {np.unique(tri_domains)}")

    return tri_domains
# def f04_convert_mphtxt_to_gmsh(msh_file):
#     nodes, tris, edgs, tri_domains = f01_load_comsol_mphtxt(mphtxt_file)
#     # Filtrează tri_domains pentru a include doar valorile asociate triunghiurilor
#     tri_domains_correct = tri_domains[4:758]  # Elimină primele 4 valori (6, 0, 755, 1)
#     if len(tri_domains_correct) != len(tris):
#         raise ValueError(f"Lungimea tri_domains_correct ({len(tri_domains_correct)}) nu coincide cu lungimea tris ({len(tris)})")
#     if not np.all(np.isin(tri_domains_correct, [1, 2, 3])):
#         raise ValueError(f"tri_domains_correct conține valori neașteptate: {np.unique(tri_domains_correct)}")

#     # Scrie fișierul .msh
#     with open(msh_file, "w") as out:
#         out.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
#         out.write("$Nodes\n")
#         out.write(f"{len(nodes)}\n")
#         for i, (x, y) in enumerate(nodes, 1):
#             out.write(f"{i} {x} {y} 0\n")
#         out.write("$EndNodes\n")
#         out.write("$Elements\n")
#         out.write(f"{len(tris) + len(edgs)}\n")
#         # Scrie muchiile
#         for i, (n1, n2) in enumerate(edgs, 1):
#             out.write(f"{i} 1 2 0 0 {n1 + 1} {n2 + 1}\n")
#         # Scrie triunghiurile
#         for i, t in enumerate(tris, len(edgs) + 1):
#             dom = tri_domains_correct[i - len(edgs)]
#             out.write(f"{i} 2 2 {dom} {dom} {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")
#         out.write("$EndElements\n")
#     print(f"Număr tri-domenii detectate: {len(np.unique(tri_domains_correct))}")
#     return tri_domains_correct

###########################################################################
############### (5) A CINCEA FUNCITE ########################################
###########################################################################

def f05_load_comsol_msh(filename):
    """
    Citește un fișier .msh (format Gmsh) și returnează nodurile, triunghiurile și muchiile.
    
    Args:
        filename (str): Calea către fișierul .msh
        
    Returns:
        tuple: (nodes, triangles, edges)
            - nodes: np.array de formă (n_nodes, 2) cu coordonatele (x, y)
            - triangles: np.array de formă (n_triangles, 3) cu indicii nodurilor triunghiurilor
            - edges: np.array de formă (n_edges, 2) cu indicii nodurilor muchiilor
    """
    try:
        # Citește mesh-ul folosind meshio
        mesh = meshio.read(filename)
        
        # Extrage nodurile (coordonate x, y)
        nodes = mesh.points[:, :2]  # Ignoră z pentru mesh 2D
        
        # Extrage triunghiurile
        triangles = np.array([], dtype=int).reshape(0, 3)
        for cell_block in mesh.cells:
            if cell_block.type == "triangle":
                triangles = cell_block.data
                break
        
        # Extrage muchiile
        edges = np.array([], dtype=int).reshape(0, 2)
        for cell_block in mesh.cells:
            if cell_block.type == "line":
                edges = cell_block.data
                break
        
        # Ajustare indexare (dacă este 1-based)
        if triangles.size > 0 and triangles.min() >= 1:
            triangles -= 1
        if edges.size > 0 and edges.min() >= 1:
            edges -= 1
        
        # print("##################################################")
        # print("Suntem in funcita: f05_load_comsol_msh()")
        # print(f"0. Successfully read {filename}")
        # print(f"1. Nodes gmsh: {nodes.shape}")
        # print(f"2. Triangles gmsh: {triangles.shape}")
        # print(f"3. Edges gmsh: {edges.shape}")
        # print("##################################################")     
        return nodes, triangles, edges
    
    except Exception as e:
        print(f"Error reading .msh file: {str(e)}")
        return np.array([]), np.array([], dtype=int).reshape(0, 3), np.array([], dtype=int).reshape(0, 2)
    


###########################################################################
############### (6) A SASEA FUNCITE ########################################
###########################################################################

def f06_convert_msh_to_xdmf(msh_file, xdmf_file=None):
    """
    Convertește un fișier .msh (COMSOL/Gmsh) în format .xdmf pentru DolfinX/FEniCS.

    Parameters
    ----------
    msh_file : str
        Numele fișierului .msh de intrare.
    xdmf_file : str, optional
        Numele fișierului .xdmf de ieșire.
        Dacă nu este specificat, se generează automat din msh_file.
    """
    if xdmf_file is None:
        xdmf_file = msh_file.replace(".msh", ".xdmf")

    try:
        mesh = meshio.read(msh_file)
        meshio.write(xdmf_file, mesh)
        print(f"✅ Successfully wrote {xdmf_file}")
        return xdmf_file
    except Exception as e:
        print(f"❌ Error reading/writing mesh: {e}")
        return None


###########################################################################
############### (7) A SASEA FUNCITE ########################################
###########################################################################

def f07_extract_triangles_to_xdmf(msh_file, xdmf_file=None):
    """
    Citește un fișier .msh și extrage doar elementele de tip triunghi,
    apoi le salvează într-un fișier .xdmf compatibil cu DolfinX/FEniCS.

    Parameters
    ----------
    msh_file : str
        Fișierul de intrare (.msh).
    xdmf_file : str, optional
        Fișierul de ieșire (.xdmf). Dacă nu este specificat, 
        numele va fi derivat din msh_file.
    """
    if xdmf_file is None:
        xdmf_file = msh_file.replace(".msh", "_triangles.xdmf")

    # Citește mesh-ul original
    msh = meshio.read(msh_file)

    # Extrage doar triunghiurile
    triangle_cells = [c for c in msh.cells if c.type == "triangle"]
    if not triangle_cells:
        raise RuntimeError("❌ The mesh does not contain any 'triangle' cells.")

    tri_cells = np.vstack([c.data for c in triangle_cells])
    print(f"✅ Number of triangles: {tri_cells.shape[0]}")

    # Filtrare cell_data pentru triunghiuri
    triangle_cell_data = {}
    for key, data in msh.cell_data_dict.items():
        if isinstance(data, dict):
            if "triangle" in data:
                vals = data["triangle"]
            else:
                continue
        elif isinstance(data, (list, tuple)):
            vals = data[0]
        else:
            vals = data

        vals = np.array(vals).astype(np.int32)
        triangle_cell_data[key] = [vals]

    # Creează mesh doar cu triunghiuri + tag-uri
    mesh_tri = meshio.Mesh(
        points=msh.points,
        cells=[("triangle", tri_cells)],
        cell_data=triangle_cell_data
    )

    # Scrie în format XDMF
    mesh_tri.write(xdmf_file, file_format="xdmf", data_format="HDF")
    print(f"✅ Filtered mesh saved to '{xdmf_file}'")

    # Test rapid: verifică conținutul fișierului
    test_mesh = meshio.read(xdmf_file)
    print("📂 Cell types in XDMF:", [c.type for c in test_mesh.cells])
    print("📂 Cell data keys:", list(test_mesh.cell_data_dict.keys()))

    if "gmsh:physical" in test_mesh.cell_data_dict:
        phys_data = test_mesh.cell_data_dict["gmsh:physical"]
        if isinstance(phys_data, (list, tuple)):
            phys_data = phys_data[0]
        if isinstance(phys_data, dict):
            phys_data = phys_data.get("triangle", phys_data)
        print("📊 Unique gmsh:physical values:", np.unique(phys_data))
    else:
        print("⚠️ No 'gmsh:physical' found in cell_data.")

    return xdmf_file


###########################################################################
############### (8) A OPTA FUNCITE ########################################
###########################################################################

def f08_inspect_xdmf_mesh(xdmf_file, mesh_name="Grid", comm=MPI.COMM_WORLD):
    """
    Citește un mesh din fișier XDMF și afișează informații
    despre topologie, geometrie și spațiul de funcții.

    Parameters
    ----------
    xdmf_file : str
        Calea către fișierul XDMF.
    mesh_name : str, optional
        Numele domeniului stocat în XDMF (default: "Grid").
    comm : MPI communicator, optional
        Implicit `MPI.COMM_WORLD`.
    """
    with io.XDMFFile(comm, xdmf_file, "r") as xdmf:
        domain = xdmf.read_mesh(name=mesh_name)

    # Creează spațiul de funcții
    V = fem.functionspace(domain, ("Lagrange", 1))

    if comm.rank == 0:
        tdim = domain.topology.dim
        gdim = domain.geometry.dim
        print("=== MESH INFO ===")
        print(f"Topological dim: {tdim}, Geometrical dim: {gdim}")
        print(f"Cells: {domain.topology.index_map(tdim).size_local}")
        print(f"Vertices: {domain.topology.index_map(0).size_local}")

        # Primele 3 noduri
        print("First 3 vertices:")
        for i, pt in enumerate(domain.geometry.x[:3]):
            print(f"  {i}: {pt}")

        print(f"Degrees of freedom: {V.dofmap.index_map.size_local}")

        el = V.element.basix_element
        print(f"Element family: {el.family}")
        print(f"Element degree: {el.degree}")
        print(f"Value shape: {el.value_shape}, rank: {len(el.value_shape)}")
        print(f"Block size: {V.dofmap.bs}")

        # Conectivitate celule -> vârfuri
        domain.topology.create_connectivity(tdim, 0)
        conn = domain.topology.connectivity(tdim, 0)
        print("Cell-to-vertex connectivity (3 cells):")
        for i in range(min(3, domain.topology.index_map(tdim).size_local)):
            print(f"  Cell {i}: {conn.links(i)}")

    return domain, V



###########################################################################
############### (9) A NOUA FUNCITE ########################################
###########################################################################

def f09_read_xdmf_mesh(xdmf_file, mesh_name="Grid", comm=MPI.COMM_WORLD, inspector=None):
    """
    Încarcă un mesh din fișier XDMF și, opțional, îl inspectează.

    Parameters
    ----------
    xdmf_file : str
        Calea către fișierul XDMF.
    mesh_name : str, optional
        Numele domeniului stocat în fișier (default: "Grid").
    comm : MPI communicator, optional
        Implicit `MPI.COMM_WORLD`.
    inspector : callable, optional
        Funcție de tip `inspector(xdmf_file)` pentru afișare info extra.

    Returns
    -------
    domain : dolfinx.mesh.Mesh
        Domeniul încărcat.
    """
    try:
        # Dacă avem funcție de inspectare, o apelăm doar pe rank 0
        if inspector is not None and comm.rank == 0:
            inspector(xdmf_file)

        with XDMFFile(comm, xdmf_file, "r") as xdmf:
            domain = xdmf.read_mesh(name=mesh_name)

            if comm.rank == 0:
                print(f"Mesh successfully read under name: {mesh_name}!")
                print(f"Number of cells (triangles): "
                      f"{domain.topology.index_map(domain.topology.dim).size_local}")

        return domain

    except Exception as e:
        if comm.rank == 0:
            print(f"Error reading XDMF file: {str(e)}")
        raise


'''
### %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ### 
### %%%%%%%%%% PLOTURI %%% GRAFICE %%% FOTOGRAFII %%%%%%%%%%%%%%%%%%%%%%%%% ### 
### %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ### 
'''


############################################################################
################ (1) PRIMUL PLOT ########################################
############################################################################
def p01_plot_mesh_mphtxt(nodes, tris, edgs, title="Fig. (1) Mesh din COMSOL .mphtxt"):
    """Desenează mesh-ul cu noduri, triunghiuri și muchii roșii"""
    plt.figure(figsize=(8, 8))

    # if tris.size > 0:
    #     plt.triplot(nodes[:, 0], nodes[:, 1], tris, color="blue", linewidth=0.5)

    if edgs.size > 0:
        for e in edgs:
            x = nodes[e, 0]
            y = nodes[e, 1]
            plt.plot(x, y, color="red")

    # plt.scatter(nodes[:, 0], nodes[:, 1], s=5, color="black")
    plt.gca().set_aspect("equal")
    plt.xlabel('x [m]', fontsize = 16)
    plt.ylabel('y [m]', fontsize = 16)
    plt.title(title, fontsize = 16)
    plt.show()

############################################################################
################ (2) AL DOILEA PLOT  ########################################
############################################################################
def p02_plot_mesh_with_labels(nodes, tris, edgs, title="Fig. (2) Mesh cu noduri numerotate"):
    """Desenează mesh-ul și numerotează nodurile"""
    plt.figure(figsize=(8, 8))

    # --- Triunghiurile albastre ---
    if tris.size > 0:
        plt.triplot(nodes[:, 0], nodes[:, 1], tris, color="green", linewidth=0.1)

    # --- Edge-urile roșii ---
    if edgs.size > 0:
        for e in edgs:
            x = nodes[e, 0]
            y = nodes[e, 1]
            plt.plot(x, y, color="red", linewidth=0.3)

    # --- Nodurile ---
    plt.scatter(nodes[:, 0], nodes[:, 1], s=2, color="yellow")

    # --- Numerotare noduri ---
    for i, (x, y) in enumerate(nodes):
        plt.text(x, y, str(i), color="black", fontsize=8,
                 ha="center", va="center")

    # --- Aspect și titlu ---
    plt.gca().set_aspect("equal")
    plt.xlabel('x [m]', fontsize = 16)
    plt.ylabel('y [m]', fontsize = 16)
    plt.title(title, fontsize = 16)
    plt.grid(True)
    plt.show()



############################################################################
################ (3) AL TREILEA PLOT  ########################################
############################################################################
# Exemplu: definim manual materialele pe domenii
# domain_materials = {
#     0: "IRON 1",
#     1: "AIR",
#     2: "COIL_1 (PLUS)",
#     3: "IRON_2",
#     4: "COIL_2 (PLUS)",
#     5: "COIL_1 (MINUS)",
#     6: "COIL_2 (MINUS)",
#     # etc... în funcție de ce ai tu în COMSOL
# }

# # Culori definite manual (în aceeași ordine ca ID-urile)
# domain_colors = {
#     0: "#808080",  # Steel → gri
#     1: "#ADD8E6",  # Air → albastru deschis
#     2: "#B87333",  # Copper → maro cupru
#     3: "#808080",  # Steel → gri
#     4: "#B87333",  # Copper → maro cupru
#     5: "#B87333",  # Copper → maro cupru
#     6: "#B87333",  # Copper → maro cupru
# }
################################################################
# def p03_plot_domains_mphtxt(nodes, tris, edgs, domains, domain_materials, domain_colors, title="Domenii cu culori controlate"):
#     num_domains = domains.max() + 1
#     print(f"Număr domenii detectate: {num_domains}")

#     # Creez o listă de culori în ordinea domeniilor
#     colors = [domain_colors[d] for d in sorted(domain_colors.keys())]
#     cmap = mcolors.ListedColormap(colors)

#     plt.figure(figsize=(8, 8))
#     tpc = plt.tripcolor(
#         nodes[:, 0], nodes[:, 1], tris,
#         facecolors=domains,
#         cmap=cmap,          # folosim cmap-ul definit manual
#         edgecolors="k",
#         linewidth=0.3
#     )

#     # Adaug muchiile roșii
#     for e in edgs:
#         x = nodes[e, 0]
#         y = nodes[e, 1]
#         plt.plot(x, y, color="red", linewidth=1.2)

#     # # Legendă cu numele materialelor
#     # handles = [plt.Rectangle((0,0),1,1, color=domain_colors[d]) for d in domain_materials]
#     # labels = [f"Domain {d}: {name}" for d, name in domain_materials.items()]
#     # plt.legend(handles, labels, loc="upper right", fontsize=10)
#     # Text pe fiecare domeniu (în centru)
#     centroids = np.array([nodes[tri].mean(axis=0) for tri in tris])
#     for dom_id in np.unique(domains):
#         # poziția medie a triunghiurilor din domeniu
#         cx, cy = centroids[domains == dom_id].mean(axis=0)
#         label = domain_materials.get(dom_id, f"Domain {dom_id}")
#         plt.text(cx, cy, label, ha="center", va="center", fontsize=18, color="white")

#     plt.gca().set_aspect("equal")
#     plt.xlabel("x [m]", fontsize=16)
#     plt.ylabel("y [m]", fontsize=16)
#     plt.title(title, fontsize=16)
#     plt.show()

################################################
def p03_plot_domains_mphtxt(nodes, tris, edgs, domains, domain_materials, domain_colors, title="Domenii cu culori controlate", domain_label_pos=None):

    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np

    num_domains = domains.max() + 1
    print(f"Număr domenii detectate: {num_domains}")

    # Creez o listă de culori în ordinea domeniilor
    colors = [domain_colors[d] for d in sorted(domain_colors.keys())]
    cmap = mcolors.ListedColormap(colors)

    plt.figure(figsize=(8, 8))
    tpc = plt.tripcolor(
        nodes[:, 0], nodes[:, 1], tris,
        facecolors=domains,
        cmap=cmap,          # folosim cmap-ul definit manual
        edgecolors="k",
        linewidth=0.3
    )

    # Adaug muchiile roșii
    for e in edgs:
        x = nodes[e, 0]
        y = nodes[e, 1]
        plt.plot(x, y, color="red", linewidth=1.2)

    # Text pe fiecare domeniu
    centroids = np.array([nodes[tri].mean(axis=0) for tri in tris])
    for dom_id in np.unique(domains):
        label = domain_materials.get(dom_id, f"Domain {dom_id}")

        # Verificăm dacă poziția e dată manual
        if domain_label_pos and dom_id in domain_label_pos:
            cx, cy = domain_label_pos[dom_id]
        else:
            # poziția medie a triunghiurilor din domeniu
            cx, cy = centroids[domains == dom_id].mean(axis=0)

        plt.text(cx, cy, label, ha="center", va="center", fontsize=18, color="white")

    plt.gca().set_aspect("equal")
    plt.xlabel("x [m]", fontsize=16)
    plt.ylabel("y [m]", fontsize=16)
    plt.title(title, fontsize=16)
    plt.show()


############################################################################
################ (4) AL PATRULEA PLOT  ########################################
############################################################################

def p04_plot_cell_tags(xdmf_file, title, cmap):
    """
    Încarcă mesh-ul și cell tags dintr-un fișier XDMF
    și face plot interactiv cu PyVista.
    """
    comm = MPI.COMM_WORLD

    # --- Încarcă mesh și meshtags din XDMF ---
    with io.XDMFFile(comm, xdmf_file, "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
        ct = xdmf.read_meshtags(mesh, name="Grid")

    # --- Creează PyVista grid ---
    tdim = mesh.topology.dim
    cells, types, x = plot.vtk_mesh(mesh)
    grid = pyvista.UnstructuredGrid(cells, types, x)

    # --- Atașează valorile meshtag-urilor ---
    num_local_cells = mesh.topology.index_map(tdim).size_local
    grid.cell_data["Marker"] = ct.values[ct.indices < num_local_cells]
    grid.set_active_scalars("Marker")

    # --- Plotare ---
    plotter = pyvista.Plotter()
    plotter.add_text("Cell tags", font_size=12, color="black")
    plotter.add_mesh(
        grid.copy(),
        scalars="Marker",
        cmap=cmap,
        show_edges=True,
        edge_color="black",
        line_width=1,
        show_scalar_bar=True,
        opacity=1.0
    )

    plotter.view_xy()
    plotter.link_views()

    # Vizualizare 2D
    plotter.view_xy()
    plotter.add_title(title)

    if not pyvista.OFF_SCREEN:
        plotter.show()
    else:
        pyvista.start_xvfb()
        plotter.screenshot("cell_tags.png", window_size=[800, 800])

############################################################################
################ (5) AL CINCELEA PLOT  ########################################
############################################################################

# def p05_plot_domains_gmesh1(nodes, triangles, edges, title="Mesh din COMSOL .msh"):
#     """
#     Plotează un mesh din noduri, triunghiuri și muchii (edge-uri).
    
#     Parameters
#     ----------
#     nodes : np.ndarray
#         Coordonatele nodurilor (N x 2).
#     triangles : np.ndarray
#         Indicii triunghiurilor (M x 3).
#     edges : np.ndarray
#         Indicii muchiilor (K x 2).
#     title : str
#         Titlul figurii (default: 'Mesh din COMSOL .msh').
#     """
#     if nodes.size == 0:
#         print("⚠️ Niciun nod în mesh. Probabil încărcarea a eșuat.")
#         return

#     print("Nodes:", nodes.shape)
#     print("Triangles:", triangles.shape)
#     print("Edges:", edges.shape)

#     plt.figure(figsize=(8, 8))

#     # --- Triunghiuri albastre ---
#     if triangles.size > 0:
#         plt.triplot(nodes[:, 0], nodes[:, 1], triangles, color="blue", linewidth=0.5)

#     # --- Muchii roșii ---
#     if edges.size > 0:
#         for e in edges:
#             x = nodes[e, 0]
#             y = nodes[e, 1]
#             plt.plot(x, y, color="red")

#     # --- Noduri ---
#     plt.scatter(nodes[:, 0], nodes[:, 1], s=5, color="black")

#     # --- Aspect ---
#     plt.gca().set_aspect("equal")
#     plt.title(title)
#     plt.show()

# def p05_plot_domains_gmesh1(nodes, triangles, edges, tri_domains=None,
#                             domain_colors=None, domain_materials=None,
#                             title="Mesh din COMSOL .msh"):
#     """
#     Plotează un mesh din noduri, triunghiuri și muchii, colorând subdomeniile.
#     Această versiune corectează automat tri_domains dacă are lungime incorectă.

#     Parameters
#     ----------
#     nodes : np.ndarray
#         Coordonatele nodurilor (N x 2).
#     triangles : np.ndarray
#         Indicii triunghiurilor (M x 3).
#     edges : np.ndarray
#         Indicii muchiilor (K x 2).
#     tri_domains : np.ndarray, optional
#         Vector care conține pentru fiecare triunghi indexul subdomeniului.
#     domain_colors : dict, optional
#         Dicționar {id_domeniu: culoare} pentru fiecare subdomeniu.
#     domain_materials : dict, optional
#         Dicționar {id_domeniu: nume_material}, pentru etichete text.
#     title : str
#         Titlul figurii.
#     """
#     if nodes.size == 0:
#         print("⚠️ Niciun nod în mesh. Probabil încărcarea a eșuat.")
#         return

#     # --- Corectare tri_domains ---
#     if tri_domains is None or len(tri_domains) != len(triangles):
#         print(f"⚠️ tri_domains incorect sau lipsă. Corectăm automat.")
#         tri_domains_corrected = np.zeros(len(triangles), dtype=int)
#         if tri_domains is not None:
#             # copiem valorile existente pe primele poziții
#             tri_domains_corrected[:len(tri_domains)] = tri_domains
#         tri_domains = tri_domains_corrected

#     plt.figure(figsize=(8, 8))

#     # --- Colorare pe subdomenii ---
#     if domain_colors is not None:
#         # Verificăm dacă toate valorile tri_domains au culoare definită
#         unique_domains = np.unique(tri_domains)
#         for dom_id in unique_domains:
#             if dom_id not in domain_colors:
#                 # dacă lipsește, atribuim o culoare default (gri)
#                 domain_colors[dom_id] = "#808080"

#         colors = [domain_colors[d] for d in sorted(domain_colors.keys())]
#         cmap = mcolors.ListedColormap(colors)

#         tpc = plt.tripcolor(
#             nodes[:, 0], nodes[:, 1], triangles,
#             facecolors=tri_domains,
#             cmap=cmap,
#             edgecolors="k",
#             linewidth=0.3
#         )
#     else:
#         # fără colorare pe subdomenii
#         plt.triplot(nodes[:, 0], nodes[:, 1], triangles, color="blue", linewidth=0.5)

#     # --- Muchii roșii ---
#     if edges.size > 0:
#         for e in edges:
#             x = nodes[e, 0]
#             y = nodes[e, 1]
#             plt.plot(x, y, color="red", linewidth=1.2)

#     # --- Noduri negre ---
#     plt.scatter(nodes[:, 0], nodes[:, 1], s=5, color="black")

#     # --- Etichete text pentru subdomenii ---
#     if domain_materials is not None:
#         centroids = np.array([nodes[tri].mean(axis=0) for tri in triangles])
#         for dom_id in np.unique(tri_domains):
#             cx, cy = centroids[tri_domains == dom_id].mean(axis=0)
#             label = domain_materials.get(dom_id, f"Domain {dom_id}")
#             plt.text(cx, cy, label, ha="center", va="center", fontsize=14, color="white")

#     # --- Aspect și titlu ---
#     plt.gca().set_aspect("equal")
#     plt.title(title, fontsize=16)
#     plt.xlabel("x [m]")
#     plt.ylabel("y [m]")
#     plt.show()

##################################

def p05_plot_domains_gmesh1(nodes, triangles, edges, tri_domains_corect=None, domain_colors=None, domain_materials=None, title="Mesh din COMSOL .msh", domain_label_pos=None):
    """
    Plotează un mesh din noduri, triunghiuri și muchii, colorând subdomeniile.
    Această versiune corectează automat tri_domains dacă are lungime incorectă.
    Permite setarea manuală a pozițiilor etichetelor pentru domenii.

    Parameters
    ----------
    nodes : np.ndarray
        Coordonatele nodurilor (N x 2).
    triangles : np.ndarray
        Indicii triunghiurilor (M x 3).
    edges : np.ndarray
        Indicii muchiilor (K x 2).
    tri_domains : np.ndarray, optional
        Vector care conține pentru fiecare triunghi indexul subdomeniului.
    domain_colors : dict, optional
        Dicționar {id_domeniu: culoare} pentru fiecare subdomeniu.
    domain_materials : dict, optional
        Dicționar {id_domeniu: nume_material}, pentru etichete text.
    title : str
        Titlul figurii.
    domain_label_pos : dict, optional
        Dicționar {id_domeniu: (cx, cy)} pentru poziții manuale ale etichetelor.
    """

    if nodes.size == 0:
        print("⚠️ Niciun nod în mesh. Probabil încărcarea a eșuat.")
        return
    num_domains = tri_domains_corect.max() 
    print(f"Număr tri-domenii detectate 1: {num_domains}")
    # --- Corectare tri_domains ---
    if tri_domains_corect is None or len(tri_domains_corect) != len(triangles):
        print(f"⚠️ tri_domains incorect sau lipsă. Corectăm automat.")
        tri_domains_corrected = np.zeros(len(triangles), dtype=int)
        if tri_domains_corect is not None:
            tri_domains_corrected[:len(tri_domains_corect)] = tri_domains_corect
        tri_domains_corect = tri_domains_corrected

    num_domains = tri_domains_corect.max() 
    print(f"Număr tri-domenii detectate 2: {num_domains}")
    plt.figure(figsize=(8, 8))

    # --- Colorare pe subdomenii ---
    if domain_colors is not None:
        unique_domains = np.unique(tri_domains_corect)
        for dom_id in unique_domains:
            if dom_id not in domain_colors:
                domain_colors[dom_id] = "#AC1717"  # gri default

        colors = [domain_colors[d] for d in sorted(domain_colors.keys())]
        cmap = mcolors.ListedColormap(colors)

        plt.tripcolor(
            nodes[:, 0], nodes[:, 1], triangles,
            facecolors=tri_domains_corect,
            cmap=cmap,
            edgecolors="k",
            linewidth=0.3
        )
    else:
        plt.triplot(nodes[:, 0], nodes[:, 1], triangles, color="blue", linewidth=0.5)

    # --- Muchii roșii ---
    if edges.size > 0:
        for e in edges:
            x = nodes[e, 0]
            y = nodes[e, 1]
            plt.plot(x, y, color="red", linewidth=1.2)

    # --- Noduri negre ---
    plt.scatter(nodes[:, 0], nodes[:, 1], s=5, color="black")

    # --- Etichete text pentru subdomenii ---
    if domain_materials is not None:
        centroids = np.array([nodes[tri].mean(axis=0) for tri in triangles])
        for dom_id in np.unique(tri_domains_corect):
            label = domain_materials.get(dom_id, f"Domain {dom_id}")
            # dacă s-a definit poziție manuală, o folosim
            if domain_label_pos and dom_id in domain_label_pos:
                cx, cy = domain_label_pos[dom_id]
            else:
                cx, cy = centroids[tri_domains_corect == dom_id].mean(axis=0)
            plt.text(cx, cy, label, ha="center", va="center", fontsize=14, color="white")

    # --- Aspect și titlu ---
    plt.gca().set_aspect("equal")
    plt.title(title, fontsize=16)
    plt.xlabel("x [m]",fontsize = 16)
    plt.ylabel("y [m]",fontsize = 16)
    plt.show()


############################################################################
################ (6) AL SASELEA PLOT  ########################################
############################################################################


def p06_visualize_xdmf_mesh(xdmf_file, mesh_name="Grid", comm=MPI.COMM_WORLD, title = "Mesh in format .xdmf (pentru FEniCS)"):
    """
    Încarcă un mesh din fișier XDMF și îl vizualizează cu PyVista (doar pe rank 0).

    Parameters
    ----------
    xdmf_file : str
        Calea către fișierul XDMF.
    mesh_name : str, optional
        Numele domeniului stocat în fișier (default: "Grid").
    comm : MPI communicator, optional
        Implicit `MPI.COMM_WORLD`.

    Returns
    -------
    domain : dolfinx.mesh.Mesh
        Domeniul încărcat.
    """
    try:
        with XDMFFile(comm, xdmf_file, "r") as xdmf:
            domain = xdmf.read_mesh(name=mesh_name)
    except Exception as e:
        print(f"Error reading XDMF file {xdmf_file}: {str(e)}")
        raise

    if comm.rank == 0:
        plotter = pv.Plotter()
        topology, cell_types, geometry = plot.vtk_mesh(domain)
        grid = pv.UnstructuredGrid(topology, cell_types, geometry)
        plotter.add_mesh(grid, show_edges=True, color="white", edge_color="blue")

        # Vizualizarea axelor și a grilei
        plotter.show_bounds(
            grid='front',
            location='outer',
            all_edges=True,
            xlabel='X (m)',
            ylabel='Y (m)',
            zlabel='Z (m)'
        )

        # Vizualizare 2D
        plotter.view_xy()
        plotter.add_title(title)
        plotter.show()

    return domain

############################################################################
################ (7) AL SAPTELEA PLOT  ########################################
############################################################################


def p07_plot_subdomains(nodes, tris, edgs, subdomains, colors, labels, figsize=(10,10)):
    """
    Plotează un mesh 2D cu subdomenii colorate și etichetate.

    Parameters
    ----------
    nodes : ndarray (N, 2)
        Coordonatele nodurilor.
    tris : ndarray (M, 3)
        Triunghiurile din mesh (indici către `nodes`).
    edgs : ndarray (K, 2)
        Muchiile de frontieră (indici către `nodes`).
    subdomains : list of lists
        Fiecare element conține indicii nodurilor care definesc un subdomeniu.
    colors : list of str
        Culorile asociate fiecărui subdomeniu.
    labels : list of str
        Etichetele pentru fiecare subdomeniu.
    figsize : tuple, optional
        Dimensiunea figurii matplotlib.
    """
    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.set_aspect("equal")

    # --- Triunghiurile ---
    plt.triplot(nodes[:,0], nodes[:,1], tris, color="blue", linewidth=0.5)

    # --- Muchiile de frontieră ---
    for e in edgs:
        x = nodes[e,0]
        y = nodes[e,1]
        plt.plot(x, y, color="red", linewidth=1)

    # --- Subdomeniile ---
    patches = []
    for sub, color in zip(subdomains, colors):
        coords = nodes[sub, :]
        if not np.array_equal(coords[0], coords[-1]):
            coords = np.vstack([coords, coords[0]])  # închide poligonul
        poly = Polygon(coords, closed=True) # matplotlib 
        patches.append(poly)

    p = PatchCollection(patches, facecolor=colors, alpha=0.3, edgecolor="k", linewidths=1.5)
    ax.add_collection(p)

    # --- Etichetele ---
    for sub, label in zip(subdomains, labels):
        coords = nodes[sub, :]
        if not np.array_equal(coords[0], coords[-1]):
            coords = np.vstack([coords, coords[0]])
        centroid = coords.mean(axis=0)
        ax.text(centroid[0], centroid[1], label, ha="center", va="center", fontsize=20, alpha=0.9)

    # --- Nodurile ---
    plt.scatter(nodes[:,0], nodes[:,1], s=10, color="black")

    plt.title("Fig. (7) Mesh with Colored Subdomains and Labels",fontsize = 16)
    plt.show()


############################################################################
################ (8) AL OPTELEA PLOT  ########################################
############################################################################
def p08_plot_external_boundary(nodes,subdomains):
    """
    Creează poligoane pentru fiecare subdomeniu,
    calculează frontiera externă + găurile interioare
    și le plotează.

    Parametri:
        nodes (np.ndarray): coordonatele nodurilor, shape (N,2)
        subdomains (list[list[int]]): liste de indici pentru fiecare subdomeniu
    """
    polygons = []
    for sub in subdomains:
        coords = nodes[sub, :]
        if not np.array_equal(coords[0], coords[-1]):
            coords = np.vstack([coords, coords[0]])
        polygons.append(Polygon1(coords))

    # unirea tuturor subdomeniilor → domeniul mare
    domain_union = unary_union(polygons)

    fig, ax = plt.subplots(figsize=(8, 8))

    # frontiera exterioară (roșu)
    ext_coords = np.array(domain_union.exterior.coords)
    ax.plot(ext_coords[:, 0], ext_coords[:, 1], 'r-', linewidth=2, label="External Boundary")

    # frontiere interioare (găuri) (albastru)
    for interior in domain_union.interiors:
        int_coords = np.array(interior.coords)
        ax.plot(int_coords[:, 0], int_coords[:, 1], 'b--', linewidth=1.5, label="Internal Boundary")

    # noduri
    ax.scatter(nodes[:, 0], nodes[:, 1], s=10, color='black', label="Nodes")

    ax.set_aspect("equal")
    ax.set_title("Fig. (8)External + Internal Boundaries of Domain",fontsize = 16)
    ax.legend()
    plt.show()

    return domain_union



###########################################################################
################ (9) AL NOUALEA PLOT  ########################################
############################################################################


def p09_plot_dirichlet_neumann_boundaries(nodes, tris, tol=1e-12, title = "Frontiere Dirichlet (roșu) și Neumann (verde)"):
    """
    Plotează frontierele Dirichlet (roșu) și Neumann (verde)
    pentru o discretizare triunghiulară.

    Parametri:
        nodes (np.ndarray): coordonatele nodurilor, shape (N,2)
        tris (np.ndarray): indici triunghiuri, shape (M,3)
        tol (float): toleranță numerică pentru y=0
    """

    # toate muchiile triunghiurilor
    edges = np.vstack([tris[:, [0,1]],
                       tris[:, [1,2]],
                       tris[:, [2,0]]])

    # sortare pentru a putea detecta duplicate
    edges = np.sort(edges, axis=1)

    # numărăm aparițiile
    edges_tuple = [tuple(e) for e in edges]
    unique_edges, counts = np.unique(edges_tuple, return_counts=True, axis=0)

    # frontieră = muchii unice
    boundary_edges = np.array([e for e, c in zip(unique_edges, counts) if c == 1])

    # condiții de frontieră
    dirichlet_edges = np.array([e for e in boundary_edges if np.all(np.abs(nodes[e,1]) < tol)])
    neumann_edges   = np.array([e for e in boundary_edges if not np.all(np.abs(nodes[e,1]) < tol)])

    # --- plot ---
    plt.figure(figsize=(8, 8))
    plt.gca().set_aspect('equal')

    # mesh
    plt.triplot(nodes[:,0], nodes[:,1], tris, color="lightblue", linewidth=0.5)

    # Dirichlet (roșu)
    for e in dirichlet_edges:
        plt.plot(nodes[e,0], nodes[e,1], 'r', linewidth=2)

    # Neumann (verde)
    for e in neumann_edges:
        plt.plot(nodes[e,0], nodes[e,1], 'g', linewidth=2)

    plt.title(title)
    plt.xlabel("X", fontsize = 16)
    plt.ylabel("Y",fontsize = 16)
    plt.show()

    return dirichlet_edges, neumann_edges



############################################################################
################ (10) AL zecelea PLOT  ########################################
############################################################################


def p10_plot_subdomains_tris(nodes, tris, cell_tags, figsize=(10,10)):
    """
    Desenează mesh-ul și colorează fiecare triunghi în funcție de subdomeniu.

    Parameters
    ----------
    nodes : ndarray (N, 2)
        Coordonatele nodurilor.
    tris : ndarray (M, 3)
        Triunghiurile (indici către `nodes`).
    cell_tags : ndarray (M,)
        Etichetele de subdomeniu pentru fiecare triunghi.
    figsize : tuple
        Dimensiunea figurii matplotlib.
    """
    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.set_aspect("equal")

    # Creăm un obiect Triangulation
    triang = mtri.Triangulation(nodes[:,0], nodes[:,1], tris)

    # Colorăm fiecare triunghi după tag
    tpc = ax.tripcolor(triang, facecolors=cell_tags, cmap="tab10", edgecolors="k", linewidth=0.2, alpha=0.8)

    # Bară de culori
    cbar = plt.colorbar(tpc, ax=ax)
    cbar.set_label("Subdomeniu")

    plt.title("Mesh colorat pe subdomenii")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()





