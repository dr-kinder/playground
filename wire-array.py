# -----------------------------------------------------------------------------
# wire-array.py
# ----------------------------------------------------------------------------- 
# This is a slight modification to 08-magnetic-fields.ipynb that places
# six wires in a rectangular grid.  It can be used as a template to place
# wires in arbitrary locations.
# -----------------------------------------------------------------------------
# The example is derived from the fenicsx tutorial at
# https://jorgensd.github.io/dolfinx-tutorial/chapter3/em.html
# ----------------------------------------------------------------------------- 

# -----------------------------------------------------------------------------
# Import libraries and packages
# ----------------------------------------------------------------------------- 
import gmsh
import dolfinx
import multiphenicsx

import dolfinx.fem
import dolfinx.io
from dolfinx.io import gmshio
import mpi4py.MPI
import numpy as np
import petsc4py.PETSc
import ufl
import multiphenicsx.io


# -----------------------------------------------------------------------------
# Define the geometry and material properties.
# ----------------------------------------------------------------------------- 
# Geometric dimension of the mesh
dim = 2

## Background
# Radius of the entire domain
r_background = 5

# Magnetic permeability (relative) of background medium
mu_background = 1


## Wires
# Radius of individual wires
r_wire = 0.1

# Location of centers of inner and outer rings of wires.
centers_inner = 0.8 
centers_outer = 1.4 

# Number of wires.
N = 6

# Magnetic permeability (relative) of wires.
mu_wire = 1

# Current density in each wire.
J0 = 1.0


## Ring
# Inner and outer radii of ring
ring_inner = 1     
ring_outer = 1.2

# Magnetic permeability (relative) of ring.
mu_ring = 10


# -----------------------------------------------------------------------------
# Concstruct the Model
# ----------------------------------------------------------------------------- 
# Create a model.
gmsh.initialize()
gmsh.model.add("mesh")

# Define the system: a large disk.
background = gmsh.model.occ.addDisk(0, 0, 0, r_background, r_background)
gmsh.model.occ.synchronize()

# Define geometry for the ring.
# outer_ring = gmsh.model.occ.addCircle(0, 0, 0, ring_outer)
# inner_ring = gmsh.model.occ.addCircle(0, 0, 0, ring_inner)
# gmsh.model.occ.addCurveLoop([outer_ring], 5)
# gmsh.model.occ.addCurveLoop([inner_ring], 6)
# ring = gmsh.model.occ.addPlaneSurface([5, 6])
# gmsh.model.occ.synchronize()

# Create two list of circular disks to represent the wires.
## Define the wires inside the ring.
# angles_in = [n * 2*np.pi/N for n in range(N)]
# wires_in = [(2,
#             gmsh.model.occ.addDisk(centers_inner * np.cos(v),
#             centers_inner * np.sin(v), 0, r_wire, r_wire))
#             for v in angles_in]
wires_in = []
wires_in.append( (2, gmsh.model.occ.addDisk(-1, -1, 0, r_wire, r_wire)) )
wires_in.append( (2, gmsh.model.occ.addDisk(-1,  0, 0, r_wire, r_wire)) )
wires_in.append( (2, gmsh.model.occ.addDisk(-1,  1, 0, r_wire, r_wire)) )

wires_out = []
wires_out.append( (2, gmsh.model.occ.addDisk( 1, -1, 0, r_wire, r_wire)) )
wires_out.append( (2, gmsh.model.occ.addDisk( 1,  0, 0, r_wire, r_wire)) )
wires_out.append( (2, gmsh.model.occ.addDisk( 1,  1, 0, r_wire, r_wire)) )

## Define the wires outside the ring.
# angles_out = [(n + 0.5) * 2*np.pi/N for n in range(N)]
# wires_out = [(2,
#             gmsh.model.occ.addDisk(centers_outer * np.cos(v),
#             centers_outer * np.sin(v), 0, r_wire, r_wire))
#             for v in angles_out]

# Update the model.
gmsh.model.occ.synchronize()

# Resolve the boundaries of the wires and ring in the background domain.
all_surfaces = []
all_surfaces.extend(wires_in)
all_surfaces.extend(wires_out)
whole_domain = gmsh.model.occ.fragment([(2, background)], all_surfaces)

# Update the model.
gmsh.model.occ.synchronize()

# Create physical markers for each object.
# Use the following markers:
# - Vacuum: 0
# - Ring: 1
# - Inner wires: $[2,3,\dots,N+1]$
# - Outer wires: $[N+2,\dots, 2\cdot N+1]$
inner_tag = 2
outer_tag = 2 + N
background_surfaces = []
other_surfaces = []

# Gmsh can compute the mass of objects and the location of their
# centers of mass.  This loop uses these properties to determine
# which object to associate grid points with.
# 
# We will use these tags to define material properties later.
for domain in whole_domain[0]:
    center = gmsh.model.occ.getCenterOfMass(domain[0], domain[1])
    mass = gmsh.model.occ.getMass(domain[0], domain[1])

    # Identify the ring by its mass.
    # Check for ring first, because center of mass is same as background.
    if np.isclose(mass, np.pi*(ring_outer**2 - ring_inner**2)):
        gmsh.model.addPhysicalGroup(domain[0], [domain[1]], tag=1)
        other_surfaces.append(domain)
    
    # Identify the background circle by its center of mass
    elif np.allclose(center, [0, 0, 0]):
        background_surfaces.append(domain[1])

    # Identify the inner wires by their centers of mass.
    # elif np.isclose(center, centers_inner):
    elif np.isclose(center[0], -1):
        gmsh.model.addPhysicalGroup(domain[0], [domain[1]], inner_tag)
        inner_tag +=1
        other_surfaces.append(domain)

    # Identify the outer wires by their center of mass.
    # elif np.isclose(np.linalg.norm(center), centers_outer):
    elif np.isclose(center[0], +1):
        gmsh.model.addPhysicalGroup(domain[0], [domain[1]], outer_tag)
        outer_tag +=1
        other_surfaces.append(domain)

# Add marker for the vacuum.
gmsh.model.addPhysicalGroup(2, background_surfaces, tag=0)

# Create mesh resolution that is fine around the wires and
# make the grid coarse further away from the ring.
gmsh.model.mesh.field.add("Distance", 1)
edges = gmsh.model.getBoundary(other_surfaces, oriented=False)
gmsh.model.mesh.field.setNumbers(1, "EdgesList", [e[1] for e in edges])
gmsh.model.mesh.field.add("Threshold", 2)
gmsh.model.mesh.field.setNumber(2, "IField", 1)
gmsh.model.mesh.field.setNumber(2, "LcMin", r_wire / 2)
gmsh.model.mesh.field.setNumber(2, "LcMax", 5 * r_wire)
gmsh.model.mesh.field.setNumber(2, "DistMin", 2 * r_wire)
gmsh.model.mesh.field.setNumber(2, "DistMax", 4 * r_wire)
gmsh.model.mesh.field.setAsBackgroundMesh(2)
gmsh.option.setNumber("Mesh.Algorithm", 7)

# Create a mesh for this system.
gmsh.model.mesh.generate(dim)

# Bring the mesh into FEniCSx.
mesh, subdomains, boundaries = dolfinx.io.gmshio.model_to_mesh(
    gmsh.model, comm=mpi4py.MPI.COMM_WORLD, rank=0, gdim=2)

gmsh.finalize()


# Plot the entire mesh.
multiphenicsx.io.plot_mesh(mesh)

# Plot the subdomains that FEniCSx has identified.
# There should only be one for this model.
multiphenicsx.io.plot_mesh_tags(subdomains)

# -----------------------------------------------------------------------------
# Finite Element Method
# ----------------------------------------------------------------------------- 
# This loop will assign material properties to each cell in our model.
# In this case, it is the relative magnetic permeability and current density.

# Define a simple function space for properties.
Q = dolfinx.fem.FunctionSpace(mesh, ("DG", 0))

# Get the list of materials.
material_tags = np.unique(subdomains.values)

# Define functions for current density and magnetic permeability.
mu = dolfinx.fem.Function(Q)
J = dolfinx.fem.Function(Q)

# Only some regions carry current. Initialize all current densities to zero.
J.x.array[:] = 0.0

# Now, cycle over all objects and assign material properties. 
for tag in material_tags:
    cells = subdomains.find(tag)
    
    # Set values for magnetic permeability.
    if tag == 0:
        # Vacuum
        mu_ = mu_background
    elif tag == 1:
        # Ring
        mu_ = mu_ring
    else:
        # Wire
        mu_ = mu_wire

    mu.x.array[cells] = np.full_like(cells, mu_, dtype=petsc4py.PETSc.ScalarType)
    
    # Set nonzero current densities.
    if tag in range(2, 2+N):
        J.x.array[cells] = np.full_like(cells, J0, dtype=petsc4py.PETSc.ScalarType)
    elif tag in range(2+N, 2*N + 2):
        J.x.array[cells] = np.full_like(cells, -J0, dtype=petsc4py.PETSc.ScalarType)


## Set up the finite element problem.
# Define trial and test functions.
V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 2))

# Define the trial and test functions.
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Create a function to store the solution.
# This is the vector potential.  A_x = A_y = 0.
A_z = dolfinx.fem.Function(V)

# Identify the domain and boundary.
D = mesh.topology.dim
Omega = dolfinx.mesh.locate_entities_boundary(mesh, D-1, lambda x: np.full(x.shape[1], True))
dOmega = dolfinx.fem.locate_dofs_topological(V, D-1, Omega)

# Force the potential to vanish on the boundary.
bc = dolfinx.fem.dirichletbc(petsc4py.PETSc.ScalarType(0), dOmega, V)

# Define the Poisson equation we are trying to solve.
a = (1 / mu) * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = 4 * ufl.pi * J * v * ufl.dx

# Define the problem.
problem = dolfinx.fem.petsc.LinearProblem(a, L, u=A_z, bcs=[bc])

# Solve the problem.
problem.solve()

# Compute the magnetic field.
W = dolfinx.fem.VectorFunctionSpace(mesh, ("CG", 2))
B = dolfinx.fem.Function(W)
B_expr = dolfinx.fem.Expression(ufl.as_vector((A_z.dx(1), -A_z.dx(0))), W.element.interpolation_points())
B.interpolate(B_expr)


# -----------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------- 
# Plot the vector potential.
multiphenicsx.io.plot_scalar_field(A_z,"Vector Potential", warp_factor=1)

# Plot the magnetic field.
multiphenicsx.io.plot_vector_field(B,"Magnetic Field", glyph_factor=0.2)
