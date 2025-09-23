def convert_mphtxt_to_gmsh(mphtxt_file, msh_file):
    nodes = []
    elements = []
    with open(mphtxt_file, "r") as f:
        lines = f.readlines()
        mode = None
        for line in lines:
            if "Nodes" in line:
                mode = "nodes"
                continue
            if "Elements" in line:
                mode = "elements"
                continue
            if mode == "nodes":
                parts = line.strip().split()
                if len(parts) == 4:  # id, x, y, z
                    nodes.append(parts)
            if mode == "elements":
                parts = line.strip().split()
                if len(parts) > 2:  # id, type, connectivity...
                    elements.append(parts)

    with open(msh_file, "w") as out:
        out.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        out.write(f"$Nodes\n{len(nodes)}\n")
        for n in nodes:
            out.write(f"{n[0]} {n[1]} {n[2]} {n[3]}\n")
        out.write("$EndNodes\n")
        out.write(f"$Elements\n{len(elements)}\n")
        for e in elements:
            out.write(" ".join(e) + "\n")
        out.write("$EndElements\n")

# # conversie mphtxt -> msh
# convert_mphtxt_to_gmsh(
#     "steerer_steerer_fenics.mphtxt",   # fișierul exportat din COMSOL
#     "steerer_steerer_fenics.msh"       # fișierul pe care îl va crea
# )



# conversie mphtxt -> msh
# convert_mphtxt_to_gmsh(
#     "comsol2dfara_spire_1pe8_vechi1.mphtxt",   # fișierul exportat din COMSOL
#     "comsol2dfara_spire_1pe8_vechi1.msh"       # fișierul pe care îl va crea
# )


# conversie mphtxt -> msh
convert_mphtxt_to_gmsh(
    "comsol2dfara_spire_toata_vechi1.mphtxt",   # fișierul exportat din COMSOL
    "comsol2dfara_spire_toata_vechi1.msh"       # fișierul pe care îl va crea
)
