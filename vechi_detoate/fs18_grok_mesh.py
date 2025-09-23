import numpy as np

# Înlocuiește cu path-ul fișierului tău complet
with open('comsol2dfara_spire_toata_vechi1_1833_1834.mphtxt', 'r') as f:
    lines = f.readlines()

# Găsește începutul coordonatelor
coord_start = next(i for i, line in enumerate(lines) if '# Mesh point coordinates' in line) + 1

# Extrage toate 1861 coord (x, y)
coords = []
for line in lines[coord_start:coord_start + 1861]:
    if line.strip():
        x, y = map(float, line.split())
        coords.append([x, y])

points = np.array(coords)

# Afișează coordonatele
print("Nod 1833 (index 1832):", points[1832])
print("Nod 1834 (index 1833):", points[1833])

#########################################################################################

import gmsh
import numpy as np
from dolfinx.io import XDMFFile
from mpi4py import MPI

# Inițializează Gmsh
gmsh.initialize()
gmsh.model.add("2d_model")

# Adaugă punctele pentru nodurile 1833 și 1834
p1833 = gmsh.model.geo.addPoint(0.23498999, 0.02978077, 0.0, 0.001)  # Mesh size = 0.001
p1834 = gmsh.model.geo.addPoint(0.23498999, 0.03475, 0.0, 0.001)

# Adaugă muchia între 1833 și 1834
edge_1833_1834 = gmsh.model.geo.addLine(p1833, p1834)

# Adaugă nodurile din DD5 și DD7 (exemplu simplificat; completează cu toate nodurile din listele tale)
# Pentru DD5 (jug sus)
DD5_nodes = [411, 470, 536, 604, 673, 729, 780, 825, 861, 895, 925, 952, 954, 981, 1022, 1066, 1126, 1186, 1199, 1258, 1312, 1359, 1402, 1438, 1470, 1501, 1533, 1566, 1600, 1632, 1667, 1699, 1700, 1709, 1744, 1775, 1801, 1823, 1824, 1802, 1821, 1822, 1807, 1783, 1784, 1793, 1826, 1818, 1832, 1833, 1834, 1815, 1803, 1778, 1760, 1749, 1715, 1679, 1646, 1624, 1591, 1558, 1525, 1491, 1457, 1423, 1383, 1340, 1291, 1337, 1338, 1335, 1285, 1228, 1166, 1104, 1077, 1136, 1137, 1190, 1191, 1193, 1192, 1188, 1129, 1123, 1062, 1015, 974, 945, 917, 885, 849, 808, 762, 710, 650, 581, 576, 577, 646, 707, 756, 784, 735, 682, 681, 678, 677, 610, 544, 481, 424, 361, 354, 288, 235, 178, 125, 122, 75, 116, 118, 117, 73, 115, 114, 65, 60, 63, 109, 162, 224, 285, 348, 349]
DD7_nodes = [114, 115, 73, 117, 118, 116, 75, 122, 125, 178, 235, 288, 354, 361, 424, 481, 544, 610, 677, 678, 681, 682, 735, 784, 756, 707, 646, 577, 576, 581, 650, 710, 762, 808, 849, 885, 917, 945, 974, 1015, 1062, 1123, 1129, 1188, 1192, 1193, 1191, 1190, 1137, 1136, 1077, 1104, 1166, 1228, 1285, 1335, 1338, 1337, 1291, 1340, 1383, 1423, 1457, 1491, 1525, 1558, 1591, 1624, 1646, 1679, 1715, 1749, 1760, 1778, 1803, 1815, 1834, 1833, 1832, 1818, 1826, 1793, 1784, 1783, 1754, 1751, 1719, 1684, 1651, 1640, 1634, 1602, 1572, 1571, 1570, 1569, 1536, 1502, 1477, 1479, 1478, 1484, 1451, 1414, 1373, 1329, 1278, 1223, 1220, 1219, 1273, 1217, 1159, 1097, 1043, 1003, 998, 1002, 1001, 1000, 999, 970, 985, 986, 987, 962, 935, 909, 878, 843, 801, 755, 702, 642, 619, 551, 488, 457, 395, 333, 320, 321, 322, 323, 325, 387, 391, 452, 507, 513, 569, 510, 454, 489, 491, 492, 490, 434, 436, 435, 375, 312, 251, 193, 192, 137, 90, 47, 27, 25, 11, 9, 8, 18, 21, 39, 79]

# Adaugă punctele pentru DD5 și DD7 (doar un subset pentru exemplu; completează cu coords din .mphtxt)
points = {}
for idx in set(DD5_nodes + DD7_nodes):
    points[idx] = gmsh.model.geo.addPoint(coords[idx-1][0], coords[idx-1][1], 0.0, 0.001)

# Creează contururile pentru DD5 și DD7 (simplificat; trebuie să definești ordinea muchiilor)
# Pentru DD5
DD5_lines = []
for i in range(len(DD5_nodes)-1):
    DD5_lines.append(gmsh.model.geo.addLine(points[DD5_nodes[i]], points[DD5_nodes[i+1]]))
DD5_lines.append(gmsh.model.geo.addLine(points[DD5_nodes[-1]], points[DD5_nodes[0]]))  # Închide conturul
DD5_loop = gmsh.model.geo.addCurveLoop(DD5_lines)
DD5_surface = gmsh.model.geo.addPlaneSurface([DD5_loop])

# Pentru DD7 (excludem DD5 pentru a evita suprapunerea)
DD7_lines = []
for i in range(len(DD7_nodes)-1):
    DD7_lines.append(gmsh.model.geo.addLine(points[DD7_nodes[i]], points[DD7_nodes[i+1]]))
DD7_lines.append(gmsh.model.geo.addLine(points[DD7_nodes[-1]], points[DD7_nodes[0]]))
DD7_loop = gmsh.model.geo.addCurveLoop(DD7_lines + [edge_1833_1834])  # Adaugă muchia 1833-1834
DD7_surface = gmsh.model.geo.addPlaneSurface([DD7_loop, DD5_loop])  # DD7 conține DD5 ca "gaură"

# Adaugă etichete fizice pentru domenii
gmsh.model.addPhysicalGroup(2, [DD5_surface], tag=5, name="DD5")
gmsh.model.addPhysicalGroup(2, [DD7_surface], tag=7, name="DD7")

# Generează mesh-ul
gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)

# Salvează mesh-ul în format .msh
gmsh.write("model_2d.msh")
gmsh.fltk.run()
gmsh.finalize()


#########################################################################################

from dolfinx.io import gmshio

# Citește mesh-ul și etichetele fizice
mesh, cell_tags, facet_tags = gmshio.read_from_msh("model_2d.msh", MPI.COMM_WORLD, gdim=2)

# Salvează în format XDMF
with XDMFFile(MPI.COMM_WORLD, "model_2d_mesh.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(cell_tags, name="CellTags")

############################################################################################
from scipy.spatial import ConvexHull
def order_nodes(nodes, coords):
    points = coords[nodes-1]
    hull = ConvexHull(points)
    return [nodes[i] for i in hull.vertices]
DD5_ordered = order_nodes(DD5_nodes, coords)
DD7_ordered = order_nodes(DD7_nodes, coords)