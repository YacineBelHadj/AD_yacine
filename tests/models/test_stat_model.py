from AD_vibration.models import StatModel

def test_Mahalanobis():
    # Test Mahalanobis distance calculation
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])
    z = np.array([1, 2, 3, 4, 5])
    assert StatModel.Mahalanobis(x, y) == 2.0
    assert StatModel.Mahalanobis(x, z) == 0.0