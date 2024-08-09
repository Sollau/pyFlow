import numpy as np
from thesis import V_ps, V_ps_TOT, T, H_fill

a0 = 0.529
a = 5.431 * a0
tau = np.array([1 / 8, 1 / 8, 1 / 8])


def test_V_ps():
    assert np.isclose(
        V_ps(np.array([np.sqrt(3) * (2 * np.pi / a), 0, 0])), -0.1121, atol=1e-4
    )
    assert np.isclose(
        V_ps(np.array([np.sqrt(8) * (2 * np.pi / a), 0, 0])), 0.0276, atol=1e-4
    )
    assert np.isclose(
        V_ps(np.array([np.sqrt(11) * (2 * np.pi / a), 0, 0])), 0.0362, atol=1e-4
    )
    assert V_ps(np.array([0, 0, 0])) == 0
    print("All V_ps tests passed!")


def test_V_ps_TOT():
    G1 = np.array([np.sqrt(3) * (2 * np.pi / a), 0, 0])
    G2 = np.array([np.sqrt(8) * (2 * np.pi / a), 0, 0])
    G3 = np.array([np.sqrt(11) * (2 * np.pi / a), 0, 0])
    G4 = np.array([8 * np.pi / a, 8 * np.pi / a, 8 * np.pi / a])
    assert np.isclose(V_ps_TOT(G1), -0.10443, atol=1e-5)
    assert np.isclose(V_ps_TOT(G2), 0.02266, atol=1e-5)
    assert np.isclose(V_ps_TOT(G3), 0.02740, atol=1e-5)
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
    n=4
    G_vectors = np.array(
        [[i, j, l] for i in range(-n, n) for j in range(-n, n) for l in range(-n, n)]
    ) * (2 * np.pi / a)  
    
    print("First 10 Gvectors: ") 
    for x in range (0,10):
        print(G_vectors[x])
        
    G = []
    for i in range(len(G_vectors)):
        for j in range(len(G_vectors)):
            G.append(G_vectors[i] - G_vectors[j])
    
    
    print("First 10 G (diffs): ")
    for y in range (0,10):
        print(G[y])
    
               
        
    k = np.array([0, 0, 0])
    H = H_fill(k)
    print("H[0, 0]:", H[0, 0])
    print("Expected T:", T(np.array([0, 0, 0]), k))
    assert H.shape == (512, 512)  # Check the shape of the Hamiltonian matrix
    assert np.isclose(
        H[0, 0], 0.0, atol=1e-5
    )  # Check the diagonal elements
    assert np.isclose(H[1, 1], T(np.array([2 * np.pi / a, 0, 0]), k), atol=1e-5)
    assert np.isclose(
        H[0, 1], V_ps_TOT(np.array([-2 * np.pi / a, 0, 0])), atol=1e-5
    )  # Check the off-diagonal elements
    print("All H_fill tests passed!")


if __name__ == "__main__":
    test_V_ps()
    test_V_ps_TOT()
    test_T()
    test_H_fill()

    print("All tests passed!")
