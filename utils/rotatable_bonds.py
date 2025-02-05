# Inspired by https://pyxtal.readthedocs.io/en/latest/_modules/pyxtal/molecule.html.

import numpy as np
from rdkit import Chem


def remove_duplicate_tas(tas_list):
    """
    Remove duplicate torsion angles from a list of torsion angle tuples.

    Args
    ----
    tas_list : list of tuples
        A list of torsion angle tuples, each containing four values:
        (atom1, atom2, atom3, atom4).

    Returns
    -------
    list of tuples: A list of unique torsion angle tuples, where duplicate angles have been removed.
    """
    tas = np.array(tas_list)
    clean_tas = []
    considered = []
    for row in tas:
        begin = row[1]
        end = row[2]
        if not (begin, end) in considered and not (end, begin) in considered:
            if begin > end:
                begin, end = end, begin
            duplicates = tas[np.logical_and(tas[:, 1] == begin, tas[:, 2] == end)]
            duplicates_reversed = tas[
                np.logical_and(tas[:, 2] == begin, tas[:, 1] == end)
            ]
            duplicates_reversed = np.flip(duplicates_reversed, axis=1)
            duplicates = np.concatenate([duplicates, duplicates_reversed], axis=0)
            assert duplicates.shape[-1] == 4
            duplicates = duplicates[
                np.where(duplicates[:, 0] == duplicates[:, 0].min())[0]
            ]
            clean_tas.append(duplicates[np.argmin(duplicates[:, 3])].tolist())
            considered.append((begin, end))
    return clean_tas

def find_backbone_dihedrals(smiles):
    """
    Identify backbone dihedrals (phi and psi) in a peptide molecule.

    Parameters:
    smiles : str
        The SMILES string representing a peptide molecule.

    Returns:
    dict: A dictionary with keys 'phi' and 'psi' containing tuples of atom indices for these angles.
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    # Backbone dihedral patterns
    phi_pattern = "[C](=[O])[N]-[C@H]-[C](=[O])"
    psi_pattern = "[N]-[C@H]-[C](=[O])-[N]"
    omega1_pattern = "[C]-[C]-[N]-[C@H]"
    omega2_pattern = "[C@H]-[C]-[N]-[C]"

    phi_matches = mol.GetSubstructMatches(Chem.MolFromSmarts(phi_pattern))
    psi_matches = mol.GetSubstructMatches(Chem.MolFromSmarts(psi_pattern))
    omega1_matches = mol.GetSubstructMatches(Chem.MolFromSmarts(omega1_pattern))
    omega2_matches = mol.GetSubstructMatches(Chem.MolFromSmarts(omega2_pattern))

    # Convert to torsion angles by extracting unique adjacent atom indices
    phi_dihedrals = [(c1, n1, ca, c2) for c1, o, n1, ca, c2, ot in phi_matches]
    psi_dihedrals = [(n1, ca, c2, n2) for n1, ca, c2, o, n2 in psi_matches]
    omega1_dihedrals = [(cw1, c1, n1, ca) for cw1, c1, n1, ca in omega1_matches]
    omega2_dihedrals = [(ca, c2, n2, cw2) for ca, c2, n2, cw2 in omega2_matches]

    return {
        'phi': phi_dihedrals,
        'psi': psi_dihedrals,
        'omega1': omega1_dihedrals,
        'omega2': omega2_dihedrals
    }


def get_rotatable_ta_list(mol):
    """
    Find unique rotatable torsion angles of a molecule. Torsion angle is given by a tuple of adjacent atoms'
    indices (atom1, atom2, atom3, atom4), where:
    - atom2 < atom3,
    - atom1 and atom4 are minimal among neighbours of atom2 and atom3 correspondingly.

    Torsion angle is considered rotatable if:
    - the bond (atom2, atom3) is a single bond,
    - none of atom2 and atom3 are adjacent to a triple bond (as the bonds near the triple bonds must be fixed),
    - atom2 and atom3 are not in the same ring.

    Args
    ----
    mol : RDKit Mol object
        A molecule for which torsion angles need to be detected.

    Returns
    -------
    list of tuples: A list of unique torsion angle tuples corresponding to rotatable bonds in the molecule.
    """
    torsion_pattern = "[*]~[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]~[*]"
    substructures = Chem.MolFromSmarts(torsion_pattern)
    torsion_angles = remove_duplicate_tas(list(mol.GetSubstructMatches(substructures)))
    return torsion_angles


def get_torsion_angles(smiles):
    """
    Find unique rotatable torsion angles of a molecule. Torsion angle is given by a tuple of adjacent atoms'
    indices (atom1, atom2, atom3, atom4), where:
    - atom2 < atom3,
    - atom1 and atom4 are minimal among neighbours of atom2 and atom3 correspondingly.

    Torsion angle is considered rotatable if:
    - the bond (atom2, atom3) is a single bond,
    - none of atom2 and atom3 are adjacent to a triple bond (as the bonds near the triple bonds must be fixed),
    - atom2 and atom3 are not in the same ring.

    Parameters:
    smiles : str
        The SMILES string representing a molecule.

    Returns:
    list of tuples: A list of unique torsion angle tuples corresponding to rotatable bonds in the molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    return get_rotatable_ta_list(mol)


def is_hydrogen_ta(mol, ta):
    """
    Simple check whether the given torsion angle is 'hydrogen torsion angle', i.e.
    it effectively influences only positions of some hydrogens in the molecule
    """

    def is_connected_to_three_hydrogens(mol, atom_id, except_id):
        atom = mol.GetAtomWithIdx(atom_id)
        neigh_numbers = []
        for n in atom.GetNeighbors():
            if n.GetIdx() != except_id:
                neigh_numbers.append(n.GetAtomicNum())
        neigh_numbers = np.array(neigh_numbers)
        return np.all(neigh_numbers == 1)

    first = is_connected_to_three_hydrogens(mol, ta[1], ta[2])
    second = is_connected_to_three_hydrogens(mol, ta[2], ta[1])
    return first or second


def has_hydrogen_tas(mol):
    tas = get_rotatable_ta_list(mol)
    hydrogen_flags = []
    for t in tas:
        hydrogen_flags.append(is_hydrogen_ta(mol, t))
    return np.any(hydrogen_flags)
