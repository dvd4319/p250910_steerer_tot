from fs15_01_nods_edg_tri_plot import (
    f01_load_comsol_mphtxt, f02_build_adjacency, f03_find_domains, f04_convert_mphtxt_to_gmsh, 
    f04_convert_mphtxt_to_gmsh_nou, f05_load_comsol_msh
)
from fs15_01_nods_edg_tri_plot import (
    p01_plot_mesh_mphtxt, p02_plot_mesh_with_labels, p03_plot_domains_mphtxt, p05_plot_domains_gmesh1
)

from dolfinx import default_scalar_type
from dolfinx.io import XDMFFile
from dolfinx.mesh import locate_entities_boundary
from dolfinx.fem import dirichletbc,Function, functionspace, locate_dofs_topological
from dolfinx.fem.petsc import LinearProblem
from dolfinx.plot import vtk_mesh
from ufl import TestFunction, TrialFunction, as_vector, dot, grad, inner, Measure 
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import ufl 
import meshio
import basix

# MPI setup
def setup_mpi():
    """
    Initializes MPI configuration.
    """
    rank = MPI.COMM_WORLD.rank
    gdim = 2
    model_rank = 0
    mesh_comm = MPI.COMM_WORLD
    MPI.COMM_WORLD.barrier()
    comm = MPI.COMM_WORLD
    return rank, gdim, model_rank, mesh_comm, comm

# Map cell tags to nodes
def map_cell_tags_to_nodes(mesh, cell_tags, values_dict):
    """
    Maps cell tag values to nodes by averaging domain values at each node.
    """
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim, 0)
    cell_to_nodes = mesh.topology.connectivity(tdim, 0)
    
    num_nodes = mesh.topology.index_map(0).size_local
    num_cells = mesh.topology.index_map(tdim).size_local
    node_values = np.zeros(num_nodes, dtype=np.float64)
    
    node_domain_counts = [[] for _ in range(num_nodes)]
    for cell_idx in range(num_cells):
        nodes = cell_to_nodes.links(cell_idx)
        domain = cell_tags.values[cell_idx]
        for node in nodes:
            node_domain_counts[node].append(domain)
    
    for node in range(num_nodes):
        if node_domain_counts[node]:
            domain = max(set(node_domain_counts[node]), key=node_domain_counts[node].count)
            node_values[node] = values_dict[domain]
    
    return node_values

# Load and plot .mphtxt
def process_mphtxt(mphtxt_file):
    """
    Loads and visualizes mesh data from a .mphtxt file.
    """
    nodes_mphtxt, tris_mphtxt, edgs_mphtxt, tri_domains_mphtxt = f01_load_comsol_mphtxt(mphtxt_file)
    print("###############################################")
    print("DATELE SUNT CULESTE DIN .mphtxt")
    print("## ======================================= ##")
    print("Triangles mphtxt:", tris_mphtxt.shape)
    print("Edges mphtxt:", edgs_mphtxt.shape)
    print("Tri domains from mphtxt:", tri_domains_mphtxt.shape)
    print("## ======================================= ##")
    print("###############################################")

    adj_mphtxt = f02_build_adjacency(tris_mphtxt, edgs_mphtxt)
    domains_mphtxt = f03_find_domains(tris_mphtxt, adj_mphtxt)

    p01_plot_mesh_mphtxt(nodes_mphtxt, tris_mphtxt, edgs_mphtxt, title="Fig. (1) Mesh: 2D section - format .mphtxt")
    p02_plot_mesh_with_labels(nodes_mphtxt, tris_mphtxt, edgs_mphtxt, title="Fig. (2) Mesh with numbered nodes and red lines indicating the domain boundaries")

    domain_materials_mphtxt = {
        0: "0 IRON",
        1: "1 AIR",
        2: "2 COPPER",
    }
    domain_colors_mphtxt = {
        0: "#646363",
        1: "#314B7A",
        2: "#B87333",
    }
    domain_label_pos_mphtxt = {
        0: (-0.15, 0.10),
        1: (-0.05, 0.03),
        2: (-0.18, 0.04)
    }

    p03_plot_domains_mphtxt(nodes_mphtxt, tris_mphtxt, edgs_mphtxt, domains_mphtxt, domain_materials_mphtxt, domain_colors_mphtxt, title="Fig. (3) domains - format .mphtxt", domain_label_pos=domain_label_pos_mphtxt)

    return nodes_mphtxt, tris_mphtxt, edgs_mphtxt, tri_domains_mphtxt

# Convert .mphtxt to .msh and plot
def convert_mphtxt_to_msh(mphtxt_file, msh_file):
    """
    Converts .mphtxt to .msh and visualizes the resulting mesh.
    """
    tri_domains_corect = f04_convert_mphtxt_to_gmsh_nou(mphtxt_file, msh_file)
    nodes_msh, triangles_msh, edges_msh = f05_load_comsol_msh(msh_file)

    print("###############################################")
    print("DATELE SUNT CULESTE DIN .msh")
    print("## ======================================= ##")
    print("Nodes msh:", nodes_msh.shape)
    print("Triangles msh:", triangles_msh.shape)
    print("Edges msh:", edges_msh.shape)
    print("Triangle correct msh: ", tri_domains_corect.shape)
    print("Triangle nr correct msh: ", len(tri_domains_corect))
    print("## ======================================= ##")
    print("###############################################")

    domain_materials_3 = {
        1: "1 IRON",
        2: "2 AIR",
        3: "3 COPPER",
    }
    domain_colors_3 = {
        1: "#726B6B",
        2: "#154A86",
        3: "#B9761E",
    }
    domain_label_pos_3 = {
        1: (-0.15, 0.10),
        2: (-0.05, 0.03),
        3: (-0.18, 0.04)
    }

    p05_plot_domains_gmesh1(nodes_msh, triangles_msh, edges_msh, tri_domains_corect, domain_colors_3, domain_materials_3,
                           title="Fig. (5) 2D section - from format NOU .msh", domain_label_pos=domain_label_pos_3)

    return nodes_msh, triangles_msh, edges_msh, tri_domains_corect

# Convert .msh to .xdmf and extract data
def convert_msh_to_xdmf(msh_file, xdmf_tri_file, xdmf_edge_file):
    """
    Converts .msh to .xdmf for triangles and edges, and extracts mesh data.
    """
    msh_to_xdmf = meshio.read(msh_file)

    edge_cells_xdmf = None
    for cell_block in msh_to_xdmf.cells:
        if cell_block.type == "line":
            edge_cells_xdmf = cell_block.data
            break

    if edge_cells_xdmf is None:
        raise ValueError("Nu s-au găsit celule de tip 'edge' în fișierul .msh")

    if edge_cells_xdmf is None:
        print("⚠️ Nu am găsit muchii în fișierul .xdmf/.msh")
    else:
        print(f"Am găsit {len(edge_cells_xdmf)} muchii")

    triangle_cells_xdmf = None
    for cell_block in msh_to_xdmf.cells:
        if cell_block.type == "triangle":
            triangle_cells_xdmf = cell_block.data
            break

    if triangle_cells_xdmf is None:
        raise ValueError("Nu s-au găsit celule de tip 'triangle' în fișierul .msh")

    print(f" lungime triangle_cells_xdmf = {len(triangle_cells_xdmf)}")

    triangle_domains_xdmf = None
    if "gmsh:physical" in msh_to_xdmf.cell_data:
        for i, cell_block in enumerate(msh_to_xdmf.cells):
            if cell_block.type == "triangle":
                triangle_domains_xdmf = msh_to_xdmf.cell_data["gmsh:physical"][i]
                break

    if triangle_domains_xdmf is None:
        raise ValueError("Nu s-au găsit date despre domenii (gmsh:physical) pentru triunghiuri")

    print("###############################################")
    print("DATELE SUNT CULESTE DIN .msh")
    print("## ======================================= ##")
    print(f"Număr triunghiuri xdmf: {triangle_cells_xdmf.shape}")
    print(f"Număr domenii xdmf: {triangle_domains_xdmf.shape}")
    print(f"Tip triangle_domains xdmf: {type(triangle_domains_xdmf)}")
    print("## ======================================= ##")
    print("###############################################")

    nodes_xdmf = msh_to_xdmf.points[:, :2]
    tris_xdmf = triangle_cells_xdmf
    tri_domains_xdmf = triangle_domains_xdmf

    domain_materials_xdmf = {
        1: "1 IRON",
        2: "2 AIR",
        3: "3 COPPER",
    }
    domain_colors_xdmf = {
        1: "#918282",
        2: "#2C76A1",
        3: "#9AA04D",
    }
    domain_label_pos_xdmf = {
        1: (-0.15, 0.10),
        2: (-0.05, 0.03),
        3: (-0.18, 0.04),
    }

    p05_plot_domains_gmesh1(
        nodes_xdmf,
        tris_xdmf,
        edge_cells_xdmf if edge_cells_xdmf is not None else np.zeros((0, 2)),
        tri_domains_xdmf,
        domain_colors_xdmf,
        domain_materials_xdmf,
        title="Fig. (X) Domenii din XDMF",
        domain_label_pos=domain_label_pos_xdmf
    )

    meshio.write(
        xdmf_tri_file,
        meshio.Mesh(
            points=msh_to_xdmf.points,
            cells=[("triangle", triangle_cells_xdmf)],
            cell_data={"gmsh:physical": [triangle_domains_xdmf.astype(np.int32)]}
        )
    )
    print(f"✅ Fișier XDMF triunghiuri scris: {xdmf_tri_file}")

    if edge_cells_xdmf is not None and len(edge_cells_xdmf) > 0:
        meshio.write(
            xdmf_edge_file,
            meshio.Mesh(
                points=msh_to_xdmf.points,
                cells=[("line", edge_cells_xdmf)],
                cell_data={"gmsh:physical": [np.zeros(len(edge_cells_xdmf), dtype=np.int32)]}
            )
        )
        print(f"✅ Fișier XDMF muchii scris: {xdmf_edge_file}")

    return nodes_xdmf, tris_xdmf, edge_cells_xdmf, tri_domains_xdmf

# Load and plot XDMF mesh
def load_and_plot_xdmf(xdmf_tri_file):
    """
    Loads and visualizes mesh from .xdmf file.
    """
    with XDMFFile(MPI.COMM_WORLD, xdmf_tri_file, "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
        ct = xdmf.read_meshtags(mesh, name="Grid")

    nodes_xdmf_1 = mesh.geometry.x[:, :2]
    triangles_xdmf_1 = mesh.topology.connectivity(mesh.topology.dim, 0).array.reshape(-1, 3)
    tri_domains_xdmf_1 = ct.values

    domain_colors_xdmf_1 = {1: "#442A2A", 2: "#78A12C", 3: "#26C4C4"}
    domain_materials_xdmf_1 = {1: "IRON", 2: "AIR", 3: "COPPER"}
    domain_label_pos_xdmf_1 = {1: (-0.15, 0.10), 2: (-0.05, 0.03), 3: (-0.18, 0.04)}

    p05_plot_domains_gmesh1(
        nodes=nodes_xdmf_1,
        triangles=triangles_xdmf_1,
        edges=np.zeros((0, 2)),
        tri_domains_corect=tri_domains_xdmf_1,
        domain_colors=domain_colors_xdmf_1,
        domain_materials=domain_materials_xdmf_1,
        title="dupa brambureala -- Mesh din XDMF",
        domain_label_pos=domain_label_pos_xdmf_1
    )

    return mesh, ct

# Plot mesh with domains using Matplotlib
def plot_mesh_with_domains(xdmf_tri_file):
    """
    Plots mesh colored by domains using Matplotlib.
    """
    with XDMFFile(MPI.COMM_WORLD, xdmf_tri_file, "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
        cell_tags = xdmf.read_meshtags(mesh, name="Grid")

    cells = mesh.topology.connectivity(mesh.topology.dim, 0).array.reshape(-1, 3)
    points = mesh.geometry.x

    colors = cell_tags.values

    plt.tripcolor(points[:, 0], points[:, 1], cells, facecolors=colors,
                  cmap="tab20", edgecolors="k", linewidth=0.2)
    plt.gca().set_aspect("equal")
    plt.colorbar(label="Domenii")
    plt.title("Mesh colorat pe domenii (din XDMF)")
    plt.show()

# Visualize mesh with PyVista
def visualize_mesh_pyvista(xdmf_tri_file):
    """
    Visualizes mesh and domains using PyVista.
    """
    with XDMFFile(MPI.COMM_WORLD, xdmf_tri_file, "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
        cell_tags = xdmf.read_meshtags(mesh, name="Grid")

    points_raw = mesh.geometry.x
    points_2d = points_raw[:, :2]
    points_3d = np.hstack([points_2d, np.zeros((points_2d.shape[0], 1))])

    tdim = mesh.topology.dim
    cells_connectivity = mesh.topology.connectivity(tdim, 0)
    cells = np.array([cells_connectivity.links(i) for i in range(mesh.topology.index_map(tdim).size_local)])

    cells_vtk = np.hstack([np.full((cells.shape[0], 1), 3), cells]).flatten()
    cell_types = np.full(cells.shape[0], pv.CellType.TRIANGLE)

    grid = pv.UnstructuredGrid(cells_vtk, cell_types, points_3d)
    grid.cell_data["domain"] = cell_tags.values

    plotter = pv.Plotter()
    plotter.add_mesh(grid, show_edges=True, scalars="domain", cmap="tab20", show_scalar_bar=True)
    plotter.add_text("Mesh colorat pe domenii - XDMF - PY-VISTA", position="upper_edge", font_size=14, color="black")
    plotter.view_xy()
    plotter.show_axes()
    plotter.view_xy()
    plotter.add_axes()
    plotter.show()

# Solve magnetic potential A_z
def solve_magnetic_potential(xdmf_tri_file):
    """
    Solves for the magnetic vector potential A_z.
    """
    with XDMFFile(MPI.COMM_WORLD, xdmf_tri_file, "r") as xdmf:
        domain_2 = xdmf.read_mesh(name="Grid")
        ct = xdmf.read_meshtags(domain_2, name="Grid")

    dx = ufl.Measure("dx", domain=domain_2, subdomain_data=ct)

    VV = functionspace(domain_2, ("CG", 1))
    uu = TrialFunction(VV)
    vv = TestFunction(VV)

    mu0 = 4.0 * np.pi * 1e-7
    mu_Fe = 4000 * mu0
    mu_val = mu_Fe
    J_val = 3.4e6

    a = (1/mu_val) * dot(grad(uu), grad(vv)) * dx(1) \
      + (1/mu0) * dot(grad(uu), grad(vv)) * dx(2) \
      + (1/mu0) * dot(grad(uu), grad(vv)) * dx(3)

    L = J_val * vv * dx(3)

    tdim = domain_2.topology.dim
    fdim = tdim - 1
    dirichlet_facets = locate_entities_boundary(domain_2, fdim, lambda x: ~np.isclose(x[1], 0.0))
    dofs = locate_dofs_topological(VV, fdim, dirichlet_facets)
    bc = dirichletbc(default_scalar_type(0), dofs, VV)

    A_z = Function(VV)
    problem = LinearProblem(a, L, u=A_z, bcs=[bc])
    problem.solve()

    with XDMFFile(MPI.COMM_WORLD, "A_z_solution.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain_2)
        xdmf.write_function(A_z)

    np.save("Az_values.npy", A_z.x.array)

    return domain_2, A_z, ct

# Visualize A_z with PyVista
def visualize_A_z(domain_2, A_z, cell_tags):
    """
    Visualizes the magnetic potential A_z and domains using PyVista.
    """
    points_raw = domain_2.geometry.x
    points_2d = points_raw[:, :2]
    points_3d = np.hstack([points_2d, np.zeros((points_2d.shape[0], 1))])

    tdim = domain_2.topology.dim
    cells_connectivity = domain_2.topology.connectivity(tdim, 0)
    cells = np.array([cells_connectivity.links(i) for i in range(domain_2.topology.index_map(tdim).size_local)])

    cells_vtk = np.hstack([np.full((cells.shape[0], 1), 3), cells]).flatten()
    cell_types = np.full(cells.shape[0], pv.CellType.TRIANGLE)

    grid = pv.UnstructuredGrid(cells_vtk, cell_types, points_3d)
    grid.cell_data["domain"] = cell_tags.values

    Az_grid = pv.UnstructuredGrid(cells_vtk, cell_types, points_3d)
    Az_grid.point_data["A_z"] = A_z.x.array
    Az_grid.set_active_scalars("A_z")
    Az_grid.cell_data["domain"] = cell_tags.values

    plotter = pv.Plotter()
    plotter.add_mesh(
        grid,
        scalars="domain",
        cmap="tab20",
        show_edges=True,
        opacity=1.0,
        show_scalar_bar=True
    )
    plotter.add_mesh(
        Az_grid,
        scalars="A_z",
        cmap="Reds",
        show_edges=True,
        scalar_bar_args={"title": "A_z"},
        opacity=0.6
    )
    plotter.view_xy()
    plotter.camera.parallel_projection = True
    plotter.show_grid()
    plotter.show()

# Compute magnetic field B
def compute_magnetic_field(xdmf_tri_file):
    """
    Computes the magnetic field B using the curl of A_z.
    """
    dim = 2
    with XDMFFile(MPI.COMM_WORLD, xdmf_tri_file, "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid")
        ct = xdmf.read_meshtags(domain, name="Grid")

    dx = ufl.Measure("dx", domain=domain, subdomain_data=ct)
    W = functionspace(domain, ("Discontinuous Lagrange", 0, (dim,)))
    B = Function(W)

    V_scalar = functionspace(domain, ("CG", 1))
    A_z = Function(V_scalar)
    A_z.x.array[:] = np.load("Az_values.npy")

    B_expr = as_vector([A_z.dx(1), -A_z.dx(0)])
    u_proj = TrialFunction(W)
    v_proj = TestFunction(W)
    a_proj = inner(u_proj, v_proj) * dx
    L_proj = inner(B_expr, v_proj) * dx

    problem_proj = LinearProblem(a_proj, L_proj, u=B)
    problem_proj.solve()

    return domain, B, ct

# Visualize magnetic field B with arrows
def visualize_magnetic_field(mesh, B, cell_tags):
    """
    Visualizes the magnetic field B with arrows using PyVista.
    """
    points_raw = mesh.geometry.x
    points_2d = points_raw[:, :2]
    points_3d = np.hstack([points_2d, np.zeros((points_2d.shape[0], 1))])

    tdim = mesh.topology.dim
    cells_connectivity = mesh.topology.connectivity(tdim, 0)
    cells = np.array([cells_connectivity.links(i) for i in range(mesh.topology.index_map(tdim).size_local)])

    cells_vtk = np.hstack([np.full((cells.shape[0], 1), 3), cells]).flatten()
    cell_types = np.full(cells.shape[0], pv.CellType.TRIANGLE)

    grid = pv.UnstructuredGrid(cells_vtk, cell_types, points_3d)
    grid.cell_data["domain"] = cell_tags.values

    B_grid = pv.UnstructuredGrid(*vtk_mesh(mesh))
    B_values = B.x.array.reshape(-1, 2)
    B_mag = np.linalg.norm(B_values, axis=1)
    B_grid.cell_data["|B|"] = B_mag
    B_grid.set_active_scalars("|B|")

    cell_centers = B_grid.cell_centers()
    vectors_cell = np.column_stack([B_values[:, 0], B_values[:, 1], np.zeros(len(B_values))])
    norms = np.linalg.norm(vectors_cell[:, :2], axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    B_norm = vectors_cell / norms
    cell_centers.cell_data.clear()
    cell_centers.cell_data["Bnorm"] = B_norm

    domain_size = max(B_grid.bounds[1] - B_grid.bounds[0], B_grid.bounds[3] - B_grid.bounds[2], 1.0)
    desired_arrow_length = 0.01 * domain_size
    quiver = cell_centers.glyph(orient="Bnorm", scale=False, factor=desired_arrow_length)

    plotter = pv.Plotter(window_size=[1200, 900])
    plotter.add_mesh(grid, scalars="domain", cmap="tab20", show_edges=True, opacity=0.0, show_scalar_bar=False)
    plotter.add_mesh(B_grid, scalars="|B|", cmap="jet", show_edges=True, opacity=1, scalar_bar_args={"title": "|B| (T)"}, show_scalar_bar=False)
    plotter.add_mesh(quiver, color="red", opacity=1, show_scalar_bar=False)

    for d in np.unique(cell_tags.values):
        mask = cell_tags.values == d
        cells_d = B_grid.extract_cells(np.where(mask)[0])
        plotter.add_mesh(cells_d, color=None, show_edges=True, edge_color="white", line_width=2, opacity=0)

    plotter.view_xy()
    plotter.camera.parallel_projection = True
    plotter.add_title("Magnetic Induction B from curl(A_z)", font_size=14)
    plotter.show_grid()
    plotter.show()

# Visualize numerical domains
def visualize_numerical_domains(xdmf_tri_file):
    """
    Visualizes numerical domain IDs using PyVista.
    """
    with XDMFFile(MPI.COMM_WORLD, xdmf_tri_file, "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
        ct = xdmf.read_meshtags(mesh, name="Grid")

    cells = mesh.topology.connectivity(mesh.topology.dim, 0).array.reshape((-1, 3))
    points = mesh.geometry.x

    if points.shape[1] == 2:
        points = np.hstack([points, np.zeros((points.shape[0], 1))])

    cell_centers = points[cells].mean(axis=1)
    cell_ids = np.arange(cells.shape[0])
    cell_domains = ct.values

    grid = pv.PolyData()
    grid.points = points
    grid.faces = np.hstack([np.full((cells.shape[0], 1), 3), cells]).astype(np.int64)

    grid.cell_data["Cell_ID"] = cell_ids
    grid.cell_data["Domain"] = cell_domains

    plotter = pv.Plotter()
    plotter.add_mesh(grid, show_edges=True, scalars="Domain", cmap="tab20")

    subset = np.arange(0, cells.shape[0], max(1, cells.shape[0] // 200))
    for i in subset:
        plotter.add_point_labels([cell_centers[i]], [f"{cell_ids[i]}:{cell_domains[i]}"],
                                 font_size=10, point_color="red", text_color="black")

    plotter.view_xy()
    plotter.camera.parallel_projection = False
    plotter.add_title("Domeniile Numerotate", font_size=14)
    plotter.show_grid()
    plotter.show()

# Print numerical values
def print_numerical_values(B, A_z, xdmf_tri_file):
    """
    Prints numerical values of A_z and B for specified cells.
    """
    with XDMFFile(MPI.COMM_WORLD, xdmf_tri_file, "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
        ct = xdmf.read_meshtags(mesh, name="Grid")

    cell_index_42 = 42
    cell_index_67 = 67
    cell_index_75 = 75
    cell_index_467 = 467

    B_values = B.x.array.reshape((-1, 2))

    print(f"B în celula {cell_index_42}:", B_values[cell_index_42])
    print(f"B în celula {cell_index_67}:", B_values[cell_index_67])
    print(f"B în celula {cell_index_75}:", B_values[cell_index_75])
    print(f"B în celula {cell_index_467}:", B_values[cell_index_467])

    B_norm = np.linalg.norm(B_values, axis=1)

    print("||B|| în celula 42:", B_norm[42])
    print("||B|| în celula 67:", B_norm[67])
    print("||B|| în celula 75:", B_norm[75])
    print("||B|| în celula 467:", B_norm[467])

    A_values = A_z.x.array
    cells = mesh.topology.connectivity(mesh.topology.dim, 0).array.reshape(-1, 3)

    def cell_average_values(cell_idx):
        nodes = cells[cell_idx]
        return np.mean(A_values[nodes])

    cell_indices = [42, 67, 75, 467]
    for idx in cell_indices:
        val = cell_average_values(idx)
        print(f"A_z mediu în celula {idx}:", val)

# Main execution
def main():
    """
    Main function to orchestrate the entire workflow.
    """
    mphtxt_file = "comsol2dfara_spire_1pe8_vechi1_3dom_403_nou.mphtxt"
    msh_file = "comsol2dfara_spire_1pe8_vechi1_3dom_403_nou.msh"
    xdmf_tri_file = msh_file.replace(".msh", "_triangles.xdmf")
    xdmf_edge_file = msh_file.replace(".msh", "_edges.xdmf")

    setup_mpi()
    process_mphtxt(mphtxt_file)
    convert_mphtxt_to_msh(mphtxt_file, msh_file)
    convert_msh_to_xdmf(msh_file, xdmf_tri_file, xdmf_edge_file)
    mesh, ct = load_and_plot_xdmf(xdmf_tri_file)
    plot_mesh_with_domains(xdmf_tri_file)
    visualize_mesh_pyvista(xdmf_tri_file)
    domain_2, A_z, ct = solve_magnetic_potential(xdmf_tri_file)
    visualize_A_z(domain_2, A_z, ct)
    domain, B, ct = compute_magnetic_field(xdmf_tri_file)
    visualize_magnetic_field(domain, B, ct)
    visualize_numerical_domains(xdmf_tri_file)
    print_numerical_values(B, A_z, xdmf_tri_file)

if __name__ == "__main__":
    main()