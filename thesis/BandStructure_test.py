import numpy as np
from BandStructure import V_ps, V_ps_TOT, T, index_generator, vector_matrix, H_fill
import scipy.constants as const

a = 5.431e-10
N = 100
tau = (1 / 8) * np.ones(3)

J_to_eV = 6.2415e18
Ry_to_eV = 13.6057039763


def test_V_ps():
    G3 = np.array([np.sqrt(3), 0, 0])
    G8 = np.array([np.sqrt(8), 0, 0])
    G11 = np.array([np.sqrt(11), 0, 0])
    G0 = np.array([0, 0, 0])

    assert np.isclose(V_ps(G3), (-0.2241 * Ry_to_eV), atol=1e-4)
    assert np.isclose(V_ps(G8), (0.0551 * Ry_to_eV), atol=1e-4)
    assert np.isclose(V_ps(G11), (0.0724 * Ry_to_eV), atol=1e-4)
    assert V_ps(G0) == 0
    print("All V_ps() tests passed!")


def test_V_ps_TOT():
    G3 = np.array([np.sqrt(3), 0, 0])
    G8 = np.array([np.sqrt(8), 0, 0])
    G11 = np.array([np.sqrt(11), 0, 0])
    G4 = np.array([4, 4, 4])

    assert np.isclose(V_ps_TOT(G3), -0.63693, atol=1e-5)
    assert np.isclose(V_ps_TOT(G8), -0.45407, atol=1e-5)
    assert np.isclose(V_ps_TOT(G11), -0.84654, atol=1e-5)
    assert np.isclose(V_ps_TOT(G4), 0.0, atol=1e-5)
    print("All V_ps_TOT() tests passed!")


def test_T():
    G0 = np.array([0, 0, 0])
    k0 = np.array([0, 0, 0])
    G1 = np.array([1, 0, 0])
    k1 = np.array([1, 0, 0])

    assert np.isclose(T(G0, k0), 0.0, atol=1e-5)
    assert np.isclose(T(G1, k0), 5.09943, atol=1e-5)
    assert np.isclose(T(G0, k1), 5.09943, atol=1e-5)
    assert np.isclose(T(G1, k1), 20.39775, atol=1e-5)
    print("All T() tests passed!")


def test_index_generator():
    states = 3

    assert index_generator(0, states) == (0, 0, 0)
    assert index_generator(1, states) == (0, 0, 1)
    assert index_generator(2, states) == (0, 1, -1)
    assert index_generator(5, states) == (1, -1, -1)
    assert index_generator(-4, states) == (0, -1, -1)
    assert index_generator(-1, states) == (0, 0, -1)

    print("All index_generator() tests passed!")


def test_vector_matrix():
    states = 3
    basis = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])

    G_matrix = vector_matrix(states=states, basis=basis)

    assert np.shape(G_matrix) == (
        27,
        27,
    )  # Check the shape of the reciprocal vectors matrix
    assert np.array_equal(
        G_matrix[0][0], np.array([-1, -1, -1])
    )  # Check diagonal element
    assert np.array_equal(
        G_matrix[0][1], np.array([-1, -1, 1])
    )  # Check off-diagonal element

    print("All vector_matrix() tests passed!")


def test_H_fill():
    states = 3
    basis = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])

    G_matrix = vector_matrix(states=states, basis=basis)
    k = np.array([0, 0, 0])

    H = H_fill(G_matrix=G_matrix, k=k)

    assert np.shape(H) == (27, 27)  # Check the shape of the Hamiltonian matrix
    assert H[0][0] == T([-1, -1, -1], k)  # Check diagonal element
    assert H[0][1] == V_ps_TOT([-1, -1, 1])  # Check off-diagonal element

    print("All H_fill() tests passed!")


if __name__ == "__main__":
    test_V_ps()
    test_V_ps_TOT()
    test_T()
    test_index_generator()
    test_vector_matrix()
    test_H_fill()

    print("All tests passed!")
