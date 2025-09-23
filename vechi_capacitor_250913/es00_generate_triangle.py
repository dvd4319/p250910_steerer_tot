'''
data: 2025 August 13 
din mesh generat de .geo scot doar triunghiurile
'''
import meshio
import numpy as np

msh_file = "es02_capacitor.msh"
xdmf_file = "es02_capacitor_triangles.xdmf"

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
