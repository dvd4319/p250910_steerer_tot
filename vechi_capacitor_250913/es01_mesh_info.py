from mpi4py import MPI
from dolfinx import fem, io
import numpy as np

comm = MPI.COMM_WORLD

# Load mesh
with io.XDMFFile(comm, "es02_capacitor_triangles.xdmf", "r") as xdmf:
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
