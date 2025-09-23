import numpy as np
import matplotlib.pyplot as plt
import mph
import dolfinx
from mpi4py import MPI
from petsc4py import PETSc
from ufl import Cell
import basix.ufl
import ufl
from dolfinx.mesh import CellType
from dolfinx import fem, io, mesh, plot
from dolfinx.io import XDMFFile
from dolfinx.fem.petsc import LinearProblem
import xml.etree.ElementTree as ET
import pyvista as pv 
import meshio
###################################################
comm = MPI.COMM_WORLD
########################################
def clean_line(line: str):
    """Elimină comentariile după # și întoarce lista de token-uri"""
    return line.split("#")[0].strip().split()

################################################
### 1.0. APLICA FUNCTIA load_comsol_mphtxt() ###
################################################
def load_comsol_mphtxt(filename):
    nodes = []
    triangles = []
    edges = []

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

        # --- Elemente edg ---
        if line.startswith("3 edg"):
            nnodes = int(clean_line(lines[i+1])[0])
            count  = int(clean_line(lines[i+2])[0])
            for k in range(count):
                parts = clean_line(lines[i+3+k])
                if len(parts) >= nnodes:
                    edges.append(list(map(int, parts[:nnodes])))
            i += 3 + count
            continue

        # --- Elemente tri ---
        if line.startswith("3 tri"):
            nnodes = int(clean_line(lines[i+1])[0])
            count  = int(clean_line(lines[i+2])[0])
            for k in range(count):
                parts = clean_line(lines[i+3+k])
                if len(parts) >= nnodes:
                    triangles.append(list(map(int, parts[:nnodes])))
            i += 3 + count
            continue

        i += 1

    return np.array(nodes), np.array(triangles, dtype=int), np.array(edges, dtype=int)


# === MAIN === # 
################################################
### 1.1. APLICA FUNCTIA load_comsol_mphtxt() ###
################################################

mphtxt_file = "comsol2dfara_spire_1pe8_vechi1.mphtxt"
nodes, tris, edgs = load_comsol_mphtxt(mphtxt_file)

print("Nodes:", nodes.shape)
print("Triangles:", tris.shape)
print("Edges:", edgs.shape)

plt.figure(figsize=(8,8))

font_litera = 16
if tris.size > 0:
    plt.triplot(nodes[:,0], nodes[:,1], tris, color="blue", linewidth=0.5)

if edgs.size > 0:
    for e in edgs:
        x = nodes[e,0]
        y = nodes[e,1]
        plt.plot(x, y, color="red")

plt.scatter(nodes[:,0], nodes[:,1], s=5, color="black")
plt.gca().set_aspect("equal")
plt.xlabel("x [m]", fontsize = font_litera)
plt.ylabel("y [m]", fontsize = font_litera)
plt.title("Mesh din COMSOL format .mphtxt pt. Python", fontsize = font_litera)
plt.show()


################ fs02_clean_msh #########################################


infile = "comsol2dfara_spire_1pe8_vechi1.msh"
outfile = "comsol2dfara_spire_1pe8_vechi1_clean.msh"


with open(infile, "r") as f_in, open(outfile, "w") as f_out:
    for line in f_in:
        # elimină tot ce vine după #
        clean = line.split("#")[0].strip()
        if clean:  # păstrează doar liniile ne-goale
            f_out.write(clean + "\n")

#####################################################
### 2.0. APLICA FUNCTIA  convert_mphtxt_to_gmsh() ###
#####################################################

def convert_mphtxt_to_gmsh(mphtxt_file, msh_file):
    # Load data from COMSOL mphtxt
    nodes, tris, edgs = load_comsol_mphtxt(mphtxt_file)
    
    # Ensure 1-based indexing for Gmsh
    if tris.min() >= 1:
        tris -= 1  # Convert to 0-based temporarily
    if edgs.min() >= 1:
        edgs -= 1  # Convert to 0-based temporarily
    tris += 1  # Convert to 1-based for Gmsh
    edgs += 1  # Convert to 1-based for Gmsh
    
    # Write Gmsh file
    with open(msh_file, "w") as out:
        # Mesh format
        out.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        
        # Nodes
        out.write(f"$Nodes\n{len(nodes)}\n")
        for i, (x, y) in enumerate(nodes, 1):
            out.write(f"{i} {x} {y} 0.0\n")
        out.write("$EndNodes\n")
        
        # Elements (triangles and edges)
        out.write(f"$Elements\n{len(tris) + len(edgs)}\n")
        element_id = 1
        # Triangles (element type 2)
        for t in tris:
            out.write(f"{element_id} 2 2 0 0 {t[0]} {t[1]} {t[2]}\n")
            element_id += 1
        # Edges (element type 1)
        for e in edgs:
            out.write(f"{element_id} 1 2 0 0 {e[0]} {e[1]}\n")
            element_id += 1
        out.write("$EndElements\n")
    
    print(f"Converted to Gmsh: {msh_file}")
    print(f"Nodes written: {len(nodes)}")
    print(f"Triangles written: {len(tris)}")
    print(f"Edges written: {len(edgs)}")

#####################################################
### 2.1. APLICA FUNCTIA  convert_mphtxt_to_gmsh() ###
#####################################################
convert_mphtxt_to_gmsh(
    "comsol2dfara_spire_1pe8_vechi1.mphtxt",
    "comsol2dfara_spire_1pe8_vechi1.msh"
)
#############################################################################

import numpy as np
import mph
import numpy as np
import meshio


def clean_line(line: str):
    """Elimină comentariile după # și întoarce lista de token-uri"""
    return line.split("#")[0].strip().split()

import numpy as np
import meshio

def load_comsol_msh(filename):
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
        
        print(f"Successfully read {filename}")
        print(f"Nodes: {nodes.shape}")
        print(f"Triangles: {triangles.shape}")
        print(f"Edges: {edges.shape}")
        
        return nodes, triangles, edges
    
    except Exception as e:
        print(f"Error reading .msh file: {str(e)}")
        return np.array([]), np.array([], dtype=int).reshape(0, 3), np.array([], dtype=int).reshape(0, 2)
# -------------------------------
# 1. Citește mesh-ul msh 
# -------------------------------

nodes, triangles, edges = load_comsol_msh("comsol2dfara_spire_1pe8_vechi1.msh")
if nodes.size > 0:
    print("Nodes:", nodes.shape)
    print("Triangles:", triangles.shape)
    print("Edges:", edges.shape)
else:
    print("Failed to load mesh from .mph file.")
# # # Dacă indexarea e 1-based în COMSOL
# if tris.min() == 1:
#     tris -= 1
# if edgs.min() == 1:
#     edgs -= 1

# # -------------------------------
# # 2. Vizualizare rapidă (opțional)
# # -------------------------------
# plt.figure()
# plt.triplot(nodes[:,0], nodes[:,1], tris, color="blue", linewidth=0.5)
# plt.scatter(nodes[:,0], nodes[:,1], s=5, color="red")
# plt.gca().set_aspect("equal")
# plt.title("pula title")
# plt.show()
if nodes.size > 0:
    plt.figure(figsize=(8, 8))
    if triangles.size > 0:
        plt.triplot(nodes[:, 0], nodes[:, 1], triangles, color="blue", linewidth=0.5)
    if edges.size > 0:
        for e in edges:
            x = nodes[e, 0]
            y = nodes[e, 1]
            plt.plot(x, y, color="red")
    plt.scatter(nodes[:, 0], nodes[:, 1], s=5, color="black")
    plt.gca().set_aspect("equal")
    plt.title("Mesh din comsol2dfara_spire_1pe8_vechi1.msh")
    plt.show()


try:
    mesh = meshio.read("comsol2dfara_spire_1pe8_vechi1.msh")
    meshio.write("comsol2dfara_spire_1pe8_vechi1.xdmf", mesh)
    print("Successfully wrote comsol2dfara_spire_1pe8_vechi1.xdmf")
except Exception as e:
    print(f"Error reading/writing mesh: {e}")

####################################################################################


msh_file = "comsol2dfara_spire_1pe8_vechi1.msh"
xdmf_file = "comsol2dfara_spire_1pe8_vechi1_triangles.xdmf"

# Read the Gmsh mesh
msh = meshio.read(msh_file)

# Extract only triangle elements
triangle_cells = [c for c in msh.cells if c.type == "triangle"]
if not triangle_cells:
    raise RuntimeError("The mesh does not contain any 'triangle' cells.")

# Concatenate all triangle elements into a single array
tri_cells = np.vstack([c.data for c in triangle_cells])

num_triangles = tri_cells.shape[0]
print(f"Number of triangles: {num_triangles}")

# Filter cell_data for triangles
triangle_cell_data = {}
for key, data in msh.cell_data_dict.items():
    # 'data' can be a dict (e.g., {'triangle': array(...)}) or a list/array
    if isinstance(data, dict):
        if "triangle" in data:
            vals = data["triangle"]
        else:
            continue
    elif isinstance(data, (list, tuple)):
        # If it's a list, we assume the first element corresponds to triangles
        vals = data[0]
    else:
        vals = data

    # Ensure the data type is int32 (recommended for FEniCS)
    vals = np.array(vals).astype(np.int32)
    triangle_cell_data[key] = [vals]

# Create a mesh containing only triangles + physical tags
mesh_tri = meshio.Mesh(
    points=msh.points,
    cells=[("triangle", tri_cells)],
    cell_data=triangle_cell_data
)

# Write the mesh to XDMF format
mesh_tri.write(xdmf_file, file_format="xdmf", data_format="HDF")

print(f"The filtered and converted mesh has been saved to '{xdmf_file}'.")

# Quick test: read XDMF file and inspect meshtags
test_mesh = meshio.read(xdmf_file)
print("Cell types in the XDMF file:", [c.type for c in test_mesh.cells])
print("Cell data keys:", list(test_mesh.cell_data_dict.keys()))

if "gmsh:physical" in test_mesh.cell_data_dict:
    phys_data = test_mesh.cell_data_dict["gmsh:physical"]
    if isinstance(phys_data, (list, tuple)):
        phys_data = phys_data[0]

    if isinstance(phys_data, dict):
        phys_data = phys_data.get("triangle", phys_data)

    print("Unique values of gmsh:physical:", np.unique(phys_data))
else:
    print("No 'gmsh:physical' found in cell_data.")

# #######################################################
# #### 2. Slide: MESH INFO #############
# #######################################################
# Load mesh
with io.XDMFFile(comm, "comsol2dfara_spire_1pe8_vechi1_triangles.xdmf", "r") as xdmf:
    domain = xdmf.read_mesh(name="Grid")

# Function space
V = fem.functionspace(domain, ("Lagrange", 1))

if comm.rank == 0:
    tdim = domain.topology.dim
    gdim = domain.geometry.dim
    print("=== MESH INFO ===")
    print(f"Topological dim: {tdim}, Geometrical dim: {gdim}")
    print(f"Cells: {domain.topology.index_map(tdim).size_local}")
    print(f"Vertices: {domain.topology.index_map(0).size_local}")
    print("First 3 vertices:")
    for i, pt in enumerate(domain.geometry.x[:3]):
        print(f"  {i}: {pt}")
    print(f"Degrees of freedom: {V.dofmap.index_map.size_local}")

    el = V.element.basix_element
    print(f"Element family: {el.family}")
    print(f"Element degree: {el.degree}")
    print(f"Value shape: {el.value_shape}, rank: {len(el.value_shape)}")
    print(f"Block size: {V.dofmap.bs}")

    domain.topology.create_connectivity(tdim, 0)
    conn = domain.topology.connectivity(tdim, 0)
    print("Cell-to-vertex connectivity (3 cells):")
    for i in range(min(3, domain.topology.index_map(tdim).size_local)):
        print(f"  Cell {i}: {conn.links(i)}")
# #######################################################
# #### 2. Slide: Inspecting the XDMF Mesh File #############
# #######################################################
# Function to inspect XDMF file
def inspect_xdmf(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        print("XDMF file structure:")
        for grid in root.findall(".//Grid"):
            grid_name = grid.get("Name")
            print(f"  Grid: {grid_name}")
            for data_item in grid.findall(".//DataItem"):
                name = data_item.get("Name", "Unnamed")
                ref = data_item.text.strip() if data_item.text else "No reference"
                print(f"    DataItem: {name}, Reference: {ref}")
    except Exception as e:
        print(f"Error inspecting XDMF: {str(e)}")

# #######################################################
# #### 3. Slide: Reading the Mesh #############
# #######################################################
# # Import mesh from file
xdmf_file = "comsol2dfara_spire_1pe8_vechi1_triangles.xdmf"
try:
    # Inspect the XDMF file
    if comm.rank == 0:
        inspect_xdmf(xdmf_file)

    with XDMFFile(comm, xdmf_file, "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid")
        if comm.rank == 0:
            print(f"Mesh successfully read under name: Grid!")
            print(f"Number of cells (triangles): {domain.topology.index_map(domain.topology.dim).size_local}")

except Exception as e:
    if comm.rank == 0:
        print(f"Error reading XDMF file: {str(e)}")
    raise

#########################################################################


# Încarcă mesh-ul din fișierul XDMF
xdmf_file = "comsol2dfara_spire_1pe8_vechi1_triangles.xdmf"
try:
    with XDMFFile(MPI.COMM_WORLD, xdmf_file, "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid")
except Exception as e:
    print(f"Error reading XDMF file {xdmf_file}: {str(e)}")
    exit(1)


if MPI.COMM_WORLD.rank == 0:
    plotter = pv.Plotter()
    topology, cell_types, geometry = plot.vtk_mesh(domain)
    grid = pv.UnstructuredGrid(topology, cell_types, geometry)
    plotter.add_mesh(grid, show_edges=True, color="white", edge_color="blue")

    # Vizualizarea axelor și a grilei pentru mesh 2D
    plotter.show_bounds(
        grid='front',       # afișează linii de grid pe fața frontală
        location='outer',   # poziționarea axelor
        all_edges=True,     # afișează toate axele
        xlabel='X (m)',
        ylabel='Y (m)',
        zlabel='Z (m)'
    )

    # Vizualizare 2D
    plotter.view_xy()
    plotter.add_title("Mesh in format .xdmf (pentru FEniCS)")
    plotter.show()




################################################################################################
###################################################################################################
#####################################################################################################
# save as apply_tags_and_bcs.py (run in same env where you have domain)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh
import dolfinx
import ufl

comm = MPI.COMM_WORLD

# ---------- CONFIG / your subdomain node-lists (0-based indices!) ----------
# NOTE: if your D lists are 1-based, subtract 1 from each index
D1 = np.array([90, 73, 71, 89], dtype=np.int32)
D2 = np.array([364, 380, 383, 401, 403, 404, 425, 453, 472, 488, 492, 491, 481, 464, 441, 433, 410, 432, 431, 430, 418, 447, 448, 449, 420, 400, 381], dtype=np.int32)
D3 = np.array([428, 426, 407, 389, 372, 353, 337, 323, 309, 294, 277, 258, 255, 256, 257, 272, 287, 304, 318, 330, 343, 361, 378, 395, 412, 435, 443, 465, 477, 479, 478, 473, 455, 428], dtype=np.int32)
D4 = np.array([404, 391, 387, 373, 356, 354, 339, 325, 311, 296, 278, 259, 241, 238, 235, 236, 210, 177, 138, 133, 101, 103, 100, 99, 108, 121, 124, 125, 123, 114, 85, 74, 62, 52, 44, 36, 25, 13, 4, 2, 1, 0, 7, 6, 5, 15, 28, 40, 51, 61, 70, 82, 98, 132, 169, 199, 228, 250, 269, 286, 303, 317, 329, 342, 358, 376, 392, 410, 433, 441, 464, 481, 491, 492, 488, 472, 453, 425], dtype=np.int32)
D5 = np.array([85, 114, 123, 125, 124, 121, 108, 99, 100, 103, 101, 133, 138, 177, 210, 236, 235, 238, 241, 259, 278, 296, 311, 325, 339, 354, 356, 373, 387, 391, 404, 403, 401, 383, 380, 364, 363, 348, 347, 333, 320, 315, 302, 295, 282, 265, 246, 222, 191, 158, 120, 90, 89, 71, 83, 86, 87, 88], dtype=np.int32)

subdomains = [D1, D2, D3, D4, D5]
material_labels = ["Air1", "Iron1", "Copper", "Air2", "Iron2"]
# map material IDs to some physical parameter e.g. mu_r
material_mu = {0: 1.0, 1: 100.0, 2: 1.0, 3: 1.0, 4: 100.0}  # example relative permeabilities

# ---------- Get dolfinx mesh 'domain' from your script (already loaded) ----------
# Assumes you already did:
# with XDMFFile(...): domain = xdmf.read_mesh(name="Grid")
# So domain exists in the environment. If not, read it here similarly.
# domain = ... (already available)

try:
    with XDMFFile(MPI.COMM_WORLD, xdmf_file, "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid")
except Exception as e:
    print(f"Error reading XDMF file {xdmf_file}: {str(e)}")
    exit(1)


# tdim = domain.topology.dim
# fdim = tdim - 1

# # ---------- Build cell centroids ----------
# # Ensure connectivity tdim -> 0 exists
# domain.topology.create_connectivity(tdim, 0)
# conn = domain.topology.connectivity(tdim, 0)

# ##################################################
# tdim = domain.topology.dim
# fdim = tdim - 1
# domain.topology.create_connectivity(fdim, tdim)
# domain.topology.create_connectivity(tdim, fdim)
# ######################################################
# from dolfinx.mesh import exterior_facet_indices

# boundary_facets = exterior_facet_indices(domain.topology)
# print("Număr fețe exterioare:", len(boundary_facets))


#########################################################

with XDMFFile(MPI.COMM_WORLD, xdmf_file, "r") as xdmf:
    domain = xdmf.read_mesh(name="Grid")

tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
domain.topology.create_connectivity(tdim, fdim)

from dolfinx.mesh import exterior_facet_indices
boundary_facets = exterior_facet_indices(domain.topology)

if MPI.COMM_WORLD.rank == 0:
    print("Număr fețe exterioare:", len(boundary_facets))

############################################################
num_cells = domain.topology.index_map(tdim).size_local
coords = domain.geometry.x  # global coordinates of geometry points (npoints x gdim)

cell_centroids = np.empty((num_cells, coords.shape[1]), dtype=np.float64)
for ci in range(num_cells):
    vert_indices = conn.links(ci)
    pts = coords[vert_indices, :coords.shape[1]]
    cell_centroids[ci, :] = np.mean(pts, axis=0)

# For point-in-polygon we only need 2D coords
cell_centroids_2d = cell_centroids[:, :2]

# ---------- Create polygon Paths for each subdomain (use domain geometry coords) ----------
# Important: subdomain node indices must match indexing in domain.geometry.x
polys = []
for sub in subdomains:
    # If your sub lists were 1-based, do sub-1 here. We assume they are 0-based.
    poly_xy = domain.geometry.x[sub, :2]  # take x,y
    # ensure closed polygon for readability
    if not np.allclose(poly_xy[0], poly_xy[-1]):
        poly_xy = np.vstack([poly_xy, poly_xy[0]])
    polys.append(Path(poly_xy))

# ---------- Assign cell markers by testing centroid in polygon ----------
cell_tags = np.zeros(num_cells, dtype=np.int32)  # default = 0 (background) or choose -1
for i, centroid in enumerate(cell_centroids_2d):
    assigned = False
    for pid, path in enumerate(polys):
        if path.contains_point(centroid):
            # tag domains by 1..N (or 10.. etc.)
            cell_tags[i] = pid + 1
            assigned = True
            break
    if not assigned:
        cell_tags[i] = 0  # outside the subdomains (if any)

# Print summary
unique, counts = np.unique(cell_tags, return_counts=True)
if comm.rank == 0:
    print("Cell tag distribution (tag:count):")
    for u,c in zip(unique, counts):
        print(u, c)

# ---------- Create Cell MeshTags (cells are dimension tdim) ----------
cells = np.arange(num_cells, dtype=np.int32)
cell_mtags = dolfinx.mesh.meshtags(domain, tdim, cells, cell_tags)

# ---------- Find boundary facets (edges) and mark Dirichlet/Neumann ----------
# Build facets -> exterior facets
domain.topology.create_connectivity(fdim, 0)
# dolfinx has a helper to get exterior facets indices
boundary_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)  # local indices of facets owned

# compute facet midpoints
facet_midpoints = dolfinx.mesh.compute_midpoints(domain, fdim, boundary_facets)  # shape (n, gdim)

# classify dirichlet facets (y approx 0 and x < 0)
tol = 1e-10
is_dir = np.logical_and(np.abs(facet_midpoints[:,1]) < 1e-8, facet_midpoints[:,0] < 0.0)
dirichlet_facets = boundary_facets[is_dir]
neumann_facets = boundary_facets[~is_dir]

# assign values, e.g., 1 = Dirichlet, 2 = Neumann
facet_values = np.zeros(len(boundary_facets), dtype=np.int32)
facet_values[is_dir] = 1
facet_values[~is_dir] = 2

# create facet MeshTags on mesh (dimension = fdim)
facet_mtags = dolfinx.mesh.meshtags(domain, fdim, boundary_facets, facet_values)

# ---------- Save tags for visualization (optional) ----------
# Write cell tags and facet tags into XDMF so you can inspect in Paraview
from dolfinx.io import XDMFFile

# with io.XDMFFile(MPI.COMM_WORLD, "mesh_with_tags.xdmf", "w") as xdmf:
#     xdmf.write_mesh(domain)
#     xdmf.write_meshtags(cell_mtags, domain.geometry)
#     xdmf.write_meshtags(facet_mtags, domain.geometry)


# with XDMFFile(domain.comm, "mesh_with_tags.xdmf", "w") as xdmf:
#     xdmf.write_mesh(domain)
#     # xdmf.write_meshtags(cell_mtags, "cell_tags")
#     xdmf.write_meshtags(cell_mtags, domain.geometry)
#     # xdmf.write_meshtags(facet_mtags, "facet_tags")
#     xdmf.write_meshtags(facet_mtags, domain.geometry)

from mpi4py import MPI
from dolfinx import io

with io.XDMFFile(MPI.COMM_WORLD, "cell_tags.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_meshtags(cell_mtags, domain.geometry)

with io.XDMFFile(MPI.COMM_WORLD, "facet_tags.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_meshtags(facet_mtags, domain.geometry)


if comm.rank == 0:
    print("Wrote mesh_with_tags.xdmf (contains 'cell_tags' and 'facet_tags').")

# ---------- Create Dirichlet BC in FEniCSx ----------
# Create function space
try:
    V = fem.functionspace(domain, ("Lagrange", 1))  # Lagrange, order 1
except TypeError:
    if comm.rank == 0:
        print("Warning: functionspace(domain, ('Lagrange', 1)) failed. Using basix.ufl.element.")
    element = basix.ufl.element("Lagrange", domain.ufl_cell().cellname(), 1)
    V = fem.functionspace(domain, element)

# find degrees of freedom on Dirichlet facets
dofs_dir = fem.locate_dofs_topological(V, fdim, dirichlet_facets)
bc_value = PETSc.ScalarType(0.0)  # example: A = 0 on y=0
bc = fem.dirichletbc(bc_value, dofs_dir, V)

# ---------- Create DG0 function with material property (per-cell) ----------
V0 = fem.functionspace(domain, ("DG", 0))   # piecewise constant per cell
mat_func = fem.Function(V0)

# we must map the cell_tags (local) to the function vector ordering
# For DG0, dof numbering corresponds to cells local ordering; we can assign directly.
mu_array = np.zeros(mat_func.x.array.size, dtype=np.float64)  # note: size = num local dofs (cells)
# set mu per cell from cell_tags
for local_ci in range(num_cells):
    tag = int(cell_tags[local_ci])
    if tag in material_mu:
        mu_array[local_ci] = material_mu[tag]
    else:
        mu_array[local_ci] = 1.0  # default

# assign
mat_func.x.array[:] = mu_array

# Save material function for inspection
with XDMFFile(domain.comm, "material_mu.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(mat_func)

if comm.rank == 0:
    print("Dirichlet BC, cell_tags and material function created and saved.")


######################################
if comm.rank == 0:
    print("Function space V:", V)
    print("Number of DOFs:", V.dofmap.index_map.size_global)
    print("Function space V0 (materials):", V0)
    print("Number of cells in material func:", V0.dofmap.index_map.size_global)

##########################################

# trial and test functions
u = ufl.TrialFunction(V)   # necunoscuta A_z
v = ufl.TestFunction(V)

# material coefficient (1/mu) – definit pe V0
nu = fem.Function(V0)   # ν = 1/μ
# exemplu: pune peste tot aer cu mu0
mu0 = 4*np.pi*1e-7
nu.x.array[:] = 1.0/mu0

# sursa (Jz) ca functie distribuita
J = fem.Function(V0)
J.x.array[:] = 0.0   # default
# de exemplu, pui sursa doar în celulele cu tag 5 (spire)
cells = np.where(cell_tags.values == 5)[0]
J.x.array[cells] = 1e6   # A/m^2 (exemplu)
