import numpy as np
from thesis import V_ps, V_ps_TOT, T

a0 = 0.529
a = 5.431 * a0
tau = np.array([1/8, 1/8, 1/8])

def test_V_ps():
    assert V_ps(np.array([np.sqrt(3) * (2 * np.pi / a), 0, 0])) == -0.1121
    assert V_ps(np.array([np.sqrt(8) * (2 * np.pi / a), 0, 0])) == 0.0276
    assert V_ps(np.array([np.sqrt(11) * (2 * np.pi / a), 0, 0])) == 0.0362
    assert V_ps(np.array([0, 0, 0])) == 0
    print("All V_ps tests passed!")

def test_V_ps_TOT():
    G1 = np.array([np.sqrt(3) * (2 * np.pi / a), 0, 0])
    G2 = np.array([np.sqrt(8) * (2 * np.pi / a), 0, 0])
    G3 = np.array([np.sqrt(11) * (2 * np.pi / a), 0, 0])
    G4 = np.array([2 * np.pi / a, 2 * np.pi / a, 2 * np.pi / a])
    
    assert np.isclose(V_ps_TOT(G1), V_ps(G1) * np.cos(np.dot(G1, tau) * np.pi / 4), atol=1e-5)
    assert np.isclose(V_ps_TOT(G2), V_ps(G2) * np.cos(np.dot(G2, tau) * np.pi / 4), atol=1e-5)
    assert np.isclose(V_ps_TOT(G3), V_ps(G3) * np.cos(np.dot(G3, tau) * np.pi / 4), atol=1e-5)
    assert np.isclose(V_ps_TOT(G4), V_ps(G4) * np.cos(np.dot(G4, tau) * np.pi / 4), atol=1e-5)
    
    print("All V_ps_TOT tests passed!")

def test_T():
    assert np.isclose(T(np.array([0, 0, 0]), np.array([0, 0, 0])), 0.0, atol=1e-5)
    assert np.isclose(T(np.array([2 * np.pi / a, 0, 0]), np.array([0, 0, 0])), 0.5 * (2 * np.pi / a)**2, atol=1e-5)
    assert np.isclose(T(np.array([0, 0, 0]), np.array([2 * np.pi / a, 0, 0])), 0.5 * (2 * np.pi / a)**2, atol=1e-5)
    assert np.isclose(T(np.array([2 * np.pi / a, 0, 0]), np.array([2 * np.pi / a, 0, 0])), 0.5 * (4 * np.pi / a)**2, atol=1e-5)
    print("All T tests passed!")

if __name__ == "__main__":
    test_V_ps()
    test_V_ps_TOT()
    test_T()
    print("All tests passed!")
