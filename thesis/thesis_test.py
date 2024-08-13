import numpy as np
from thesis import V_ps, V_ps_TOT, T, H_fill

a0 = 0.529  # Bohr radius (in Å)
a = 5.431 * a0  # Lattice constant of Silicon
tau = np.array([1, 1, 1]) * (a / 8)  # Primitive vector
conv = 27.2114  # Conversion from Hartree to eV


def test_V_ps():
    assert np.isclose(
        V_ps(np.array([np.sqrt(3) * (2 * np.pi / a), 0, 0])), -0.1121 * conv, atol=1e-4
    )
    assert np.isclose(
        V_ps(np.array([np.sqrt(8) * (2 * np.pi / a), 0, 0])), 0.0276 * conv, atol=1e-4
    )
    assert np.isclose(
        V_ps(np.array([np.sqrt(11) * (2 * np.pi / a), 0, 0])), 0.0362 * conv, atol=1e-4
    )
    assert V_ps(np.array([0, 0, 0])) == 0
    print("All V_ps tests passed!")


def test_V_ps_TOT():
    G1 = np.array([np.sqrt(3) * (2 * np.pi / a), 0, 0])
    G2 = np.array([np.sqrt(8) * (2 * np.pi / a), 0, 0])
    G3 = np.array([np.sqrt(11) * (2 * np.pi / a), 0, 0])
    G4 = np.array([8 * np.pi / a, 8 * np.pi / a, 8 * np.pi / a])

    assert np.isclose(V_ps_TOT(G1), -0.63721, atol=1e-5)
    assert np.isclose(V_ps_TOT(G2), -0.45490, atol=1e-5)
    assert np.isclose(V_ps_TOT(G3), -0.84654, atol=1e-5)
    assert np.isclose(V_ps_TOT(G4), 0.0, atol=1e-5)
    print("All V_ps_TOT tests passed!")


def test_T():
    assert np.isclose(T(np.array([0, 0, 0]), np.array([0, 0, 0])), 0.0, atol=1e-5)
    assert np.isclose(
        T(np.array([2 * np.pi / a, 0, 0]), np.array([0, 0, 0])), 2.39143, atol=1e-5
    )
    assert np.isclose(
        T(np.array([0, 0, 0]), np.array([2 * np.pi / a, 0, 0])), 2.39143, atol=1e-5
    )
    assert np.isclose(
        T(np.array([2 * np.pi / a, 0, 0]), np.array([2 * np.pi / a, 0, 0])),
        9.56574,
        atol=1e-5,
    )
    print("All T tests passed!")


def test_H_fill():
    n = 4
    G_vectors = np.array(
        [[i, j, l] for i in range(-n, n) for j in range(-n, n) for l in range(-n, n)]
    ) * (2 * np.pi / a)

    k = np.array([0, 0, 0])
    H = H_fill(k)
    
    assert H.shape == (512, 512)  # Check the shape of the Hamiltonian matrix
    assert np.isclose(
        H[0, 0], T(G_vectors[0], k), atol=1e-5
    )  # Check the diagonal elements
    assert np.isclose(H[1, 1], T(G_vectors[1], k), atol=1e-5)
    assert np.isclose(
        H[0, 1], V_ps_TOT(G_vectors[0] - G_vectors[1]), atol=1e-5
    )  # Check the off-diagonal elements
    print("All H_fill tests passed!")


if __name__ == "__main__":
    test_V_ps()
    test_V_ps_TOT()
    test_T()
    test_H_fill()

    print("All tests passed!")
