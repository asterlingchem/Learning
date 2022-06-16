import math
import numpy as np
from scipy import special
from scipy import linalg
import matplotlib.pyplot as plt

"""
Hartree Fock code, using the 'Nickel and Copper' Youtube video series (https://www.youtube.com/watch?v=eDAfpQIMde0)
and Szabo and Ostlund.

The Hartree Fock Roothaan equation is as follows:

1. FC = eSC
F is the Fock matrix (F = T + V_ne + V_ee)
C is the coefficient matrix (rows denote AOs, columns denote MOs, so the matrix tells us which AOs form each MO)
e is the energy eigenvalue matrix
S is the AO overlap matrix (integral of each basis function with all other basis functions) 

T = kinetic energy matrix (has the same dimensionality of S --> n(AOs) x n(AOs))
V_ne = nuclear-electron potential energy matrix (attractive)
V_ee = electron-electron potential energy matrix (repulsive)

We will start with H2, and then perhaps extend the code to H2O to learn how to use orbital angular momentum with 
Gaussian basis sets

First, set the nuclear charges and coordinates, the basis set, SCF convergence parameters, and job/printing flags.
"""

Z = [1.0, 1.0]  # charge on each nucleus in molecule
number_occupied_orbitals = 1  # restricted HF
atom_coordinates = [np.array([0.0, 0.0, 0.0]), np.array([1.4, 0.0, 0.0])]  # define coords for single calculation
basis = "STO-3G"  # STO-3G and 6-31G implemented
scf_parameters = [1e-6, 20]  # first term is the SCF energy change tolerance, second term is max number of SCF cycles
print_all_matrices = False  # print all intermediate matrices
check_scf_convergence = False  # generate plot of SCF energy as a function of step
do_single_point = False  # evaluate electronic energy at geometry specified in atom_coordinates
do_h2_scan = True  # do scan of H2 bond length

"""
Fetch the Gaussian basis set information from the Basis Set Exchange (https://www.basissetexchange.org) for H,
for example for the STO-3G basis set.

STO-3G is a contracted basis set, in which the 1s Slater function is mimicked by a fixed linear combination of 
three Gaussian functions: phi_STO-3G(1s) = N * coeff_a exp(-alpha_a * r) + coeff_b exp(-alpha_b * r) + 
coeff_c exp(-alpha_c * r), where N is a normalisation constant, the alphas controls the width of each Gaussian, and
the coeffs define the linear combination of Gaussians that make up the approximate STO.
"""

"""
Define a class for the basis functions. This involves listing each property of the basis function, including
alpha, coeff, coordinates, and angular momenta as mentioned above. For s orbitals, the angular momenta are all zero,
so we will ignore them for now. Normalisation of the orbital is achieved in self.A
"""


class PrimitiveGaussian():
    def __init__(self, alpha, coeff, coordinates, l1, l2, l3):
        self.alpha = alpha
        self.coeff = coeff
        self.coordinates = np.array(coordinates)
        self.A = (2.0 * alpha / math.pi) ** 0.75  # normalisation of s orbital, currently hard coded but to be changed


"""
Define function to calculate overlap matrix for a given basis of AOs. We need to first unpack all the primitives from
the contracted GTOs, then calculate the overlap of each primitive with all others. The way the code is currently written
is horribly inefficient, but is useful to explain the concepts. We are also ignoring orbital angular momentum here, 
since this example is for H2. We use the Gaussian product theorem (for s orbitals), which says that the product of two 
Gaussians is a Gaussian centred at a point lying between the positions of the two Gaussians. The general form of this 
theorem takes the form (see p. 142 of https://chemistlibrary.files.wordpress.com/2015/02/modern-quantum-chemistry.pdf 
for the typeset notation and derivation):

s_ij = N * coeff_i * coeff_j * exp(-q * Q_sq) * (pi / p) ** (3 / 2)

where:
N is the product of individual normalisation constants
p = alpha_i + alpha_j
q = alpha_i * alpha_j
Q_sq = (coords_i - coords_j) ** 2
coeff_i are the coordinates of the centre of Gaussian i
"""


def overlap(molecule):
    nbasis = len(molecule)  # true for this minimal basis example

    s_matrix = np.zeros([nbasis, nbasis])  # create empty matrix with dimensions of nbasis x nbasis (i.e. n(AO) x n(AO))

    for i in range(nbasis):  # loop over rows of AOs in s
        for j in range(nbasis):  # loop over columns of AOs in s
            n_primitives_basis_i = len(molecule[i])  # fetch number of primitives located on atom i in the molecule
            n_primitives_basis_j = len(molecule[j])  # fetch number of primitives located on atom j in the molecule

            for k in range(n_primitives_basis_i):
                for l in range(n_primitives_basis_j):
                    N = molecule[i][k].A * molecule[j][l].A  # normalisation constant
                    c1c2 = molecule[i][k].coeff * molecule[j][l].coeff  # product of GTO coefficients
                    p = molecule[i][k].alpha + molecule[j][l].alpha
                    q = molecule[i][k].alpha * molecule[j][l].alpha / p
                    Q = molecule[i][k].coordinates - molecule[j][l].coordinates
                    Q_sq = np.dot(Q, Q)

                    s_matrix[i, j] += N * c1c2 * np.exp(-q * Q_sq) * (np.pi / p) ** (1.5)
                    # summation over primitive overlap integrals for each pair of indices in the overlap matrix means
                    # we use the += iterater

    return s_matrix


"""
Define function to calculate kinetic energy matrix. NEED TO FIND GOOD REFERENCE FOR DERIVATION OF INTEGRAL.

Function:
(phi_i|T|phi_j) = N * coeff_i * coeff_j * (pi / p) ** 3/2 x (q) x (exp(-q x Q_sq) x (3 - (2q)(Q_sq)))
which is equivalent to
(phi_i|T|phi_j) = s_ij * (3 - (2 * q * Q_sq))

(following definitions of p, q and Q from the overlap function)
"""


def kinetic(molecule):
    nbasis = len(molecule)  # true for this minimal basis example

    kinetic_matrix = np.zeros([nbasis, nbasis])  # create empty matrix with dimensions of nbasis x nbasis
    # (i.e. n(AO) x n(AO))

    for i in range(nbasis):  # loop over rows of AOs in s
        for j in range(nbasis):  # loop over columns of AOs in s
            n_primitives_basis_i = len(molecule[i])  # fetch number of primitives located on atom i in the molecule
            n_primitives_basis_j = len(molecule[j])  # fetch number of primitives located on atom j in the molecule

            for k in range(n_primitives_basis_i):
                for l in range(n_primitives_basis_j):
                    N = molecule[i][k].A * molecule[j][l].A  # normalisation constant
                    c1c2 = molecule[i][k].coeff * molecule[j][l].coeff  # product of GTO coefficients
                    p = molecule[i][k].alpha + molecule[j][l].alpha
                    q = molecule[i][k].alpha * molecule[j][l].alpha / p
                    Q = molecule[i][k].coordinates - molecule[j][l].coordinates
                    Q_sq = np.dot(Q, Q)

                    P = (molecule[i][k].alpha * molecule[i][k].coordinates) + \
                        (molecule[j][l].alpha * molecule[j][l].coordinates)
                    Pp = P / p  # coords of new Gaussian centre
                    PG = Pp - molecule[j][
                        l].coordinates  # distance between new Gaussian centre and second Gaussian centre

                    # extract individual Cartesian coords of each PG and square the coords - useful later
                    PG_x_sq = PG[0] * PG[0]
                    PG_y_sq = PG[1] * PG[1]
                    PG_z_sq = PG[2] * PG[2]

                    # formulation described in Youtube video - likely useful when introducing orbital angular momentum
                    # s = N * c1c2 * np.exp(-q * Q_sq) * (np.pi / p) ** 1.5
                    # kinetic_matrix[i, j] += 3 * molecule[j][l].alpha * s
                    # kinetic_matrix[i, j] -= 2 * molecule[j][l].alpha * molecule[j][l].alpha * s * (PG_x_sq + 0.5/p)
                    # kinetic_matrix[i, j] -= 2 * molecule[j][l].alpha * molecule[j][l].alpha * s * (PG_y_sq + 0.5/p)
                    # kinetic_matrix[i, j] -= 2 * molecule[j][l].alpha * molecule[j][l].alpha * s * (PG_z_sq + 0.5/p)

                    # formulation described in Szabo and Ostlund (p. 412) - gives same matrix as above
                    kinetic_matrix[i, j] += q * (3 - (2 * q * Q_sq)) * (np.pi / p) ** 1.5 * np.exp(-q * Q_sq) * N * c1c2

    return kinetic_matrix


"""
Define function to calculate the nuclear-electron potential, defined on p. 415 of Szabo and Ostlund as:

(phi_1|V_ne|phi_2) = -2*pi/p * Z * exp(-q*Q_sq) * F[p|R_C - R_B|**2]

where F[t] = 0.5 * (pi / t)**0.5 * erf(t**0.5)
i.e. F[p|R_C - R_B|**2] = 0.5 * (pi / p|R_C - R_B|**2)**0.5 * erf((p|R_C - R_B|**2)**0.5)

Additional loop over atoms in molecule required.

The Youtube video suggests using the Boys function, which may be helpful for higher orbital angular momentum
later on, so the function is defined here. For H2, we can also use the erf function as shown in Szabo and Ostlund.
For both functions, it is necessary to define the behaviour of the function at t = 0 (which would correspond to the
potential when the electron and nucleus are found at the exact same position and would lead to an infinity due to 
division by 0). I am a little unclear on why the functions are defined in the way that they are.
"""


def boys(x, n):
    if x == 0:
        return 1.0 / (2 * n + 1)
    else:
        return special.gammainc(n + 0.5, x) * special.gamma(n + 0.5) * (1.0 / (2 * x ** (n + 0.5)))


def erf_fix(t):
    if t == 0:
        return 1.0
    return 0.5 * (np.pi / t) ** 0.5 * math.erf(t ** 0.5)


def ne_potential(molecule, atom_coords, nuclear_charge):
    natoms = len(nuclear_charge)
    nbasis = len(molecule)  # true for this minimal basis example
    ne_pot_matrix = np.zeros([nbasis, nbasis])

    for atom in range(natoms):  # calculate nuclear-electron attraction for each nucleus sequentially
        for i in range(nbasis):
            for j in range(nbasis):
                n_primitives_basis_i = len(molecule[i])  # fetch number of primitives located on atom i in the molecule
                n_primitives_basis_j = len(molecule[j])  # fetch number of primitives located on atom j in the molecule

                for k in range(n_primitives_basis_i):
                    for l in range(n_primitives_basis_j):
                        N = molecule[i][k].A * molecule[j][l].A  # normalisation constant
                        c1c2 = molecule[i][k].coeff * molecule[j][l].coeff  # product of GTO coefficients
                        p = molecule[i][k].alpha + molecule[j][l].alpha
                        q = molecule[i][k].alpha * molecule[j][l].alpha / p
                        Q = molecule[i][k].coordinates - molecule[j][l].coordinates
                        Q_sq = np.dot(Q, Q)

                        P = (molecule[i][k].alpha * molecule[i][k].coordinates) + \
                            (molecule[j][l].alpha * molecule[j][l].coordinates)
                        Pp = P / p  # coords of new Gaussian centre
                        PG = Pp - atom_coords[atom]  # distance between new Gaussian centre and nucleus of atom
                        PG_sq = np.dot(PG, PG)
                        t = p * PG_sq

                        # erf procedure
                        ne_pot_matrix[i, j] += N * c1c2 * (-2.0 * np.pi / p) * nuclear_charge[atom] * \
                                               np.exp(-q * Q_sq) * erf_fix(t)

                        # Boys procedure
                        # ne_pot_matrix[i, j] += -nuclear_charge[atom] * N * c1c2 * np.exp(-q * Q_sq) * \
                        # (2.0 * np.pi / p) * boys(p * PG_sq, 0)

    return ne_pot_matrix


"""
Define function to calculate 2-electron repulsion integrals. The position of each electron is described by the product
of two Gaussians (i, j or k, l) centred at nuclei A, B or C, D. This results in a 4x4 matrix (rank-4 tensor) as we need
to iterate over all possible combinations of pairs of products of Gaussians to completely describe the repulsion of each
electron with all others. As for the nuclear-electron attraction integral, the presence of an inverse function (1/r) in
the integral necessitates the use of a Fourier transform to make the integral computationally tractable. 

The 2-electron repulsion integral is described by the function:

(AB|CD) = (phi_1 phi_2 | V_ee | phi_3 phi_4)
        = 2pi**(5/2) / ((alpha_1 + alpha_2)(alpha_3 + alpha_4)(alpha_1 + alpha_2 + alpha_3 + alpha_4)**0.5
            * exp(((-(alpha_1 + alpha_2)/(alpha_1 * alpha_2)) * |R_A - R_B|^2) - 
            ((-(alpha_3 + alpha_4)/(alpha_3 * alpha_4)) * |R_C - R_D|^2))
            * F[((alpha_1 + alpha_2)(alpha_3 + (alpha_4)/(alpha_1 + alpha_2 + alpha_3 + alpha_4)) * |R_P - R_Q|**2]

where F(x) = 0.5 * (pi / x)**0.5 * erf((x)**0.5)
"""


def ee_potential(molecule):
    nbasis = len(molecule)
    ee_pot_matrix = np.zeros([nbasis, nbasis, nbasis, nbasis])

    for i in range(nbasis):  # iterate over each atom i in the molecule
        for j in range(nbasis):  # iterate over each atom j in the molecule (since we need cross-terms)
            for k in range(nbasis):  # iterate over each atom k in the molecule
                for l in range(nbasis): # iterate over each atom l in the molecule
                    n_primitives_basis_i = len(molecule[i])  # unpack the primitives from each atom i
                    n_primitives_basis_j = len(molecule[j])  # unpack the primitives from each atom j
                    n_primitives_basis_k = len(molecule[k])  # unpack the primitives from each atom k
                    n_primitives_basis_l = len(molecule[l])  # unpack the primitives from each atom l

                    for ii in range(n_primitives_basis_i):  # iterate over each primitive ii on atom i
                        for jj in range(n_primitives_basis_j):  # iterate over each primitive jj on atom j
                            for kk in range(n_primitives_basis_k):  # iterate over each primitive kk on atom k
                                for ll in range(n_primitives_basis_l):  # iterate over each primitive ll on atom l
                                    N = molecule[i][ii].A * molecule[j][jj].A  * molecule[k][kk].A * molecule[l][ll].A
                                    c1c2c3c4 = molecule[i][ii].coeff * molecule[j][jj].coeff * molecule[k][kk].coeff * \
                                               molecule[l][ll].coeff
                                    a = molecule[i][ii].alpha
                                    b = molecule[j][jj].alpha
                                    c = molecule[k][kk].alpha
                                    d = molecule[l][ll].alpha
                                    RA = molecule[i][ii].coordinates
                                    RB = molecule[j][jj].coordinates
                                    RC = molecule[k][kk].coordinates
                                    RD = molecule[l][ll].coordinates
                                    RAB = RA - RB
                                    RAB2 = np.dot(RAB, RAB)  # squared distance between initial Gaussian centres ii and jj
                                    RCD = RC - RD
                                    RCD2 = np.dot(RCD, RCD)  # squared distance between initial Gaussian centres kk and ll
                                    RP = ((a * RA) + (b * RB)) / (a + b)
                                    RQ = ((c * RC) + (d * RD)) / (c + d)
                                    RPQ = RP - RQ
                                    RPQ2 = np.dot(RPQ, RPQ)  # squared distance between new Gaussian centres

                                    func1 = 1 / ((a + b) * (c + d) * ((a + b + c + d) ** 0.5))
                                    func2 = math.exp((((- a * b) / (a + b)) * RAB2) - ((( c * d) / (c + d)) * RCD2))
                                    t = ((a + b) * (c + d) / (a + b + c + d)) * RPQ2
                                    func3 = erf_fix(t)
                                    func3_alt = boys(t, 0)

                                    # using erf
                                    ee_pot_matrix[i, j, k, l] += N * c1c2c3c4 * 2 * (np.pi) ** (5/2) * func1 * func2 * \
                                                                 func3

                                    # using boys
                                    # ee_pot_matrix[i, j, k, l] += N * c1c2c3c4 * 2 * (np.pi) ** (
                                    #             5 / 2) * func1 * func2 * func3_alt

    return ee_pot_matrix


"""
Define function to calculate classical nuclear-nuclear repulsion
"""


def nn_potential(atom_coords, nuclear_charge):
    natoms = len(nuclear_charge)
    nuclear_repulsion = 0.0

    for atom_i in range(natoms):
        for atom_j in range(natoms):
            if atom_j > atom_i:
                Zi = nuclear_charge[atom_i]
                Zj = nuclear_charge[atom_j]
                RA = atom_coords[atom_i]
                RB = atom_coords[atom_j]
                RARB = RA - RB
                RARB_norm = (np.dot(RARB, RARB)) ** 0.5
                nuclear_repulsion += (Zi * Zj) / RARB_norm

    return nuclear_repulsion


"""
Create function to calculate 2-electron term (G)
"""


def compute_G(density_matrix, ee_repulsion_matrix):

    nbasis_functions = density_matrix.shape[0]
    G = np.zeros([nbasis_functions, nbasis_functions])

    #  loop over each of the 4 indices in the two-electron repulsion matrix
    for i in range(nbasis_functions):
        for j in range(nbasis_functions):
            for k in range(nbasis_functions):
                for l in range(nbasis_functions):
                    J = ee_repulsion_matrix[i, j, k, l]
                    K = ee_repulsion_matrix[i, l, k, j]
                    G[i, j] += density_matrix[k, l] * (J - 0.5*K)  # restricted HF with spatial orbitals
                    # G is in the basis of MOs, hence the need for the density matrix P to convert AO integrals to MO basis

    return G


"""
Create function to compute new density matrix

P = occ*C*C_dagger
"""


def compute_density_matrix(mos):

    nbasis_functions = mos.shape[0]
    density_matrix = np.zeros([nbasis_functions, nbasis_functions])
    occupation = 2.0

    for i in range(nbasis_functions):  # loop over rows of density matrix
        for j in range(nbasis_functions):  # loop over columns of density matrix
            for oo in range(number_occupied_orbitals):
                C = mos[i, oo]  # rows of C matrix are AOs (hence loop over i basis functions), columns are occupied MOs
                C_dagger = mos[j, oo]  # assuming coefficients are real
                density_matrix[i, j] += occupation * C * C_dagger

    return density_matrix


"""
Create function to compute electronic energy expectation value

E_HF = P * (SUM_i(h_ii) + SUM_ij(2J_ij - K_ij) [sums run over half number of occupied orbitals]
     = P * (H_core + 0.5*G)
     
"""


def compute_electronic_energy_expectation_value(density_matrix, kinetic_matrix, ne_pot_matrix, G):

    electronic_energy_expectation = 0.0
    nbasis_functions = density_matrix.shape[0]

    h_core = kinetic_matrix + ne_pot_matrix

    for i in range(nbasis_functions):  # loop over rows of density matrix
        for j in range(nbasis_functions):  # loop over columns of density matrix
            electronic_energy_expectation += density_matrix[i, j] * (h_core[i, j] + 0.5 * G[i, j])

    return electronic_energy_expectation


"""
Create function to calculate SCF energy
"""


def scf_energy(overlap_mat, KE_mat, Vne_mat, Vee_mat, scf_params, molecule):

    electronic_energy = 0.0
    scf_tolerance, max_scf_cycles = scf_params
    nbasis_functions = len(molecule)
    density_mat = np.zeros([nbasis_functions, nbasis_functions])  # initial density matrix guess
    scf_progress = np.empty([0, 2])

    # 1. Enter into SCF cycle
    for scf_step in range(max_scf_cycles):

        print(f"SCF cycle number {scf_step + 1}\n")
        electronic_energy_old = electronic_energy

        # 2. Compute 2-electron term (G) using previous density matrix, add it to 1-electron term
        G = compute_G(density_mat, Vee_mat)
        if print_all_matrices:
            print(f"G = {G}\n")

        # 3. Form F, make S unit, get eigenvalues and eigenvectors, transform eigenvectors back (w/o unit S)
        F = KE_mat + Vne_mat + G
        if print_all_matrices:
            print(f"Fock matrix:\n {F}\n")

        # We need to make S unit, by computing S^{-1/2} S S^{-1/2} (symmetric orthogonalisation) – and we must do the
        # same operation to F to make F'
        S_inverse = linalg.inv(overlap_mat)
        S_inverse_sqrt = linalg.sqrtm(S_inverse)
        SFS = np.dot(S_inverse_sqrt, np.dot(F, S_inverse_sqrt))
        eigenvalues, eigenvectors = linalg.eigh(SFS)
        molecular_orbitals = np.dot(S_inverse_sqrt, eigenvectors)
        if print_all_matrices:
            print(f"Molecular orbitals: f{molecular_orbitals}\n")

        # 4. Form new density matrix using MOs

        density_mat = compute_density_matrix(molecular_orbitals)
        if print_all_matrices:
            print(f"Density matrix:\n {density_mat}\n")

        # 5. Compute electronic energy expectation value

        electronic_energy = compute_electronic_energy_expectation_value(density_mat, KE_mat, Vne_mat, G)
        print(f"Electronic energy: {electronic_energy}\n")

        # 6. Check convergence

        energy_diff = electronic_energy - electronic_energy_old
        print(f"∆E = {energy_diff} Ha\n")


        scf_info = np.array([scf_step + 1, electronic_energy])
        scf_progress = np.append(scf_progress, [scf_info], axis=0)

        if abs(energy_diff) < scf_tolerance:
            print(f"Convergence achieved in {scf_step + 1} steps\n")
            return electronic_energy, scf_progress

        print("----------------------------------------------")

    print("Convergence not achieved")
    return electronic_energy


"""
Define function to run a full SCF procedure
"""


def run_scf_procedure(molecule, atom_coords, nuclear_charge, scf_params):

    overlap_mat = overlap(molecule)
    kinetic_matrix = kinetic(molecule)
    ne_pot_matrix = ne_potential(molecule, atom_coords, nuclear_charge)
    ee_pot_matrix = ee_potential(molecule)
    nuclear_repulsion = nn_potential(atom_coords, nuclear_charge)
    electronic_energy, scf_progress = scf_energy(overlap_mat, kinetic_matrix, ne_pot_matrix, ee_pot_matrix, scf_params, molecule)
    total_energy = electronic_energy + nuclear_repulsion
    print(f"Total energy = {total_energy} Ha\n")
    print("----------------------------------------------")

    return electronic_energy, scf_progress, total_energy, nuclear_repulsion


def matrix_printing(molecule, atom_coords, nuclear_charge, basis_set):

    print(f"H2 {basis_set} calculation:\n")
    print(f"Overlap matrix (S):\n {overlap(molecule)}\n")
    print(f"Kinetic Energy matrix (T):\n {kinetic(molecule)}\n")
    print(f"Nuclear-Electronic Potential Energy matrix (V_ne):\n {ne_potential(molecule, atom_coords, nuclear_charge)}\n")
    print(f"Electron-Electron Potential Energy matrix (V_ee):\n {ee_potential(molecule)}\n")
    print(f"Nuclear-Nuclear Potential Energy (V_nn):\n {nn_potential(atom_coords, nuclear_charge)}\n")
    print("\n")

    return None


"""
Define the orbitals on H1 using the STO-3G basis
"""

H1_pg1a_sto = PrimitiveGaussian(0.3425250914E+01, 0.1543289673E+00, atom_coordinates[0], 0, 0, 0)
H1_pg1b_sto = PrimitiveGaussian(0.6239137298E+00, 0.5353281423E+00, atom_coordinates[0], 0, 0, 0)
H1_pg1c_sto = PrimitiveGaussian(0.1688554040E+00, 0.4446345422E+00, atom_coordinates[0], 0, 0, 0)

"""
Define the orbitals on H2 using the STO-3G basis, setting the distance between H1 and H2 as 1.2 (Angstroms)
"""

H2_pg1a_sto = PrimitiveGaussian(0.3425250914E+01, 0.1543289673E+00, atom_coordinates[1], 0, 0, 0)
H2_pg1b_sto = PrimitiveGaussian(0.6239137298E+00, 0.5353281423E+00, atom_coordinates[1], 0, 0, 0)
H2_pg1c_sto = PrimitiveGaussian(0.1688554040E+00, 0.4446345422E+00, atom_coordinates[1], 0, 0, 0)

"""
Define the STO-3G 1s orbitals on H1 and H2 using lists of the primitive Gaussians
"""

H1_1s_sto = [H1_pg1a_sto, H1_pg1b_sto, H1_pg1c_sto]  # define the H1_1s orbital using the list of the GTOs
H2_1s_sto = [H2_pg1a_sto, H2_pg1b_sto, H2_pg1c_sto]  # define the H1_1s orbital using the list of the GTOs

"""
Define the orbitals on H1 using the 6-31G basis
"""

H1_pg1a_pople = PrimitiveGaussian(0.1873113696E+02, 0.3349460434E-01, atom_coordinates[0], 0, 0, 0)
H1_pg1b_pople = PrimitiveGaussian(0.2825394365E+01, 0.2347269535E+00, atom_coordinates[0], 0, 0, 0)
H1_pg1c_pople = PrimitiveGaussian(0.6401216923E+00, 0.8137573261E+00, atom_coordinates[0], 0, 0, 0)
H1_pg2a_pople = PrimitiveGaussian(0.1612777588E+00, 1.0000000, atom_coordinates[0], 0, 0, 0)

"""
Define the orbitals on H2 using the 6-31G basis
"""

H2_pg1a_pople = PrimitiveGaussian(0.1873113696E+02, 0.3349460434E-01, atom_coordinates[1], 0, 0, 0)
H2_pg1b_pople = PrimitiveGaussian(0.2825394365E+01, 0.2347269535E+00, atom_coordinates[1], 0, 0, 0)
H2_pg1c_pople = PrimitiveGaussian(0.6401216923E+00, 0.8137573261E+00, atom_coordinates[1], 0, 0, 0)
H2_pg2a_pople = PrimitiveGaussian(0.1612777588E+00, 1.0000000, atom_coordinates[1], 0, 0, 0)

"""
Define the 6-31G 1s and 2s orbitals on H1 and H2 using lists of the primitive Gaussians
"""

H1_1s_pople = [H1_pg1a_pople, H1_pg1b_pople, H1_pg1c_pople]  # define the H1_1s orbital using the list of the GTOs
H1_2s_pople = [H1_pg2a_pople]
H2_1s_pople = [H2_pg1a_pople, H2_pg1b_pople, H2_pg1c_pople]  # define the H1_1s orbital using the list of the GTOs
H2_2s_pople = [H2_pg2a_pople]

"""
Define molecule as a list of contracted basis functions
"""

molecule_sto = [H1_1s_sto, H2_1s_sto]  # initialise a list of STO-3G AOs for the atoms in the molecule
molecule_pople = [H1_1s_pople, H1_2s_pople, H2_1s_pople, H2_2s_pople]  # initialise a list of 6-31G AOs for the atoms
# in the molecule

if print_all_matrices:
    if basis == "STO-3G":
        matrix_printing(molecule_sto, atom_coordinates, Z, basis)

    if basis == "6-31G":
        matrix_printing(molecule_pople, atom_coordinates, Z, basis)

"""
Calculate electronic energy of H2 with STO-3G or 6-31G basis
"""
if do_single_point:
    print("Single point electronic energy evaluation using user-specified geometry")
    if basis == "STO-3G":
        print(f"Level of theory: RHF/{basis}")
        electronic_energy, scf_convergence, total_energy, nuclear_repulsion = run_scf_procedure(molecule_sto,
                                                                                                atom_coordinates,
                                                                                                Z,
                                                                                                scf_parameters)

    elif basis == "6-31G":
        print(f"Level of theory: RHF/{basis}")
        electronic_energy, scf_convergence, total_energy, nuclear_repulsion = run_scf_procedure(molecule_pople,
                                                                                                atom_coordinates,
                                                                                                Z,
                                                                                                scf_parameters)

    else:
        print("Basis set not in library")

"""
Check SCF convergence as a function of step
"""

if check_scf_convergence:
    plt.xlabel("SCF step")
    plt.ylabel("Change in electronic energy / Ha")
    scf_convergence = np.delete(scf_convergence, 0, 0).T
    x = scf_convergence[0]
    y = scf_convergence[1] - scf_convergence[1][0]
    plt.plot(x, y)
    plt.show()

"""
Perform bond length scan
"""

if do_h2_scan:
    scan_data = np.empty([0, 4])

    for z_coord in np.arange(0.5, 6.5, 0.1):
        count = 1
        atom_coordinates = [np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, z_coord])]

        if basis == "STO-3G":

            H1_pg1a_sto = PrimitiveGaussian(0.3425250914E+01, 0.1543289673E+00, atom_coordinates[0], 0, 0, 0)
            H1_pg1b_sto = PrimitiveGaussian(0.6239137298E+00, 0.5353281423E+00, atom_coordinates[0], 0, 0, 0)
            H1_pg1c_sto = PrimitiveGaussian(0.1688554040E+00, 0.4446345422E+00, atom_coordinates[0], 0, 0, 0)
            H2_pg1a_sto = PrimitiveGaussian(0.3425250914E+01, 0.1543289673E+00, atom_coordinates[1], 0, 0, 0)
            H2_pg1b_sto = PrimitiveGaussian(0.6239137298E+00, 0.5353281423E+00, atom_coordinates[1], 0, 0, 0)
            H2_pg1c_sto = PrimitiveGaussian(0.1688554040E+00, 0.4446345422E+00, atom_coordinates[1], 0, 0, 0)

            H1_1s_sto = [H1_pg1a_sto, H1_pg1b_sto, H1_pg1c_sto]
            H2_1s_sto = [H2_pg1a_sto, H2_pg1b_sto, H2_pg1c_sto]

            molecule_scan = [H1_1s_sto, H2_1s_sto]

        if basis == "6-31G":

            H1_pg1a_pople = PrimitiveGaussian(0.1873113696E+02, 0.3349460434E-01, atom_coordinates[0], 0, 0, 0)
            H1_pg1b_pople = PrimitiveGaussian(0.2825394365E+01, 0.2347269535E+00, atom_coordinates[0], 0, 0, 0)
            H1_pg1c_pople = PrimitiveGaussian(0.6401216923E+00, 0.8137573261E+00, atom_coordinates[0], 0, 0, 0)
            H1_pg2a_pople = PrimitiveGaussian(0.1612777588E+00, 1.0000000, atom_coordinates[0], 0, 0, 0)
            H2_pg1a_pople = PrimitiveGaussian(0.1873113696E+02, 0.3349460434E-01, atom_coordinates[1], 0, 0, 0)
            H2_pg1b_pople = PrimitiveGaussian(0.2825394365E+01, 0.2347269535E+00, atom_coordinates[1], 0, 0, 0)
            H2_pg1c_pople = PrimitiveGaussian(0.6401216923E+00, 0.8137573261E+00, atom_coordinates[1], 0, 0, 0)
            H2_pg2a_pople = PrimitiveGaussian(0.1612777588E+00, 1.0000000, atom_coordinates[1], 0, 0, 0)

            H1_1s_pople = [H1_pg1a_pople, H1_pg1b_pople, H1_pg1c_pople]
            H1_2s_pople = [H1_pg2a_pople]
            H2_1s_pople = [H2_pg1a_pople, H2_pg1b_pople, H2_pg1c_pople]
            H2_2s_pople = [H2_pg2a_pople]

            molecule_scan = [H1_1s_pople, H1_2s_pople, H2_1s_pople, H2_2s_pople]

        print(f"Scan step {count}: Bond length {np.round(z_coord, decimals=2)} Bohr")
        print("----------------------------------------------\n")
        count += 1
        electronic_energy, scf_convergence, total_energy, nuclear_repulsion = run_scf_procedure(molecule_scan,
                                                                                                atom_coordinates,
                                                                                                Z,
                                                                                                scf_parameters)

        scan_data = np.append(scan_data,
                              [np.array([z_coord, total_energy, electronic_energy, nuclear_repulsion])],
                              axis=0)

    scan_data = scan_data.T
    bond_length = scan_data[0] * 0.529
    scan_total_energy = scan_data[1]
    scan_electronic_energy = scan_data[2]
    scan_nuclear_repulsion = scan_data[3]

    plt.xlabel("Bond distance / Å")
    plt.ylabel("HF energy / Ha")
    plt.title(f"H2 bond extension, RHF/{basis}")

    plt.plot(bond_length, scan_total_energy, label="Total energy")
    plt.plot(bond_length, scan_electronic_energy, label="Electronic energy")
    plt.plot(bond_length, scan_nuclear_repulsion, label="Nuclear repulsion")
    plt.legend()
    plt.show()
