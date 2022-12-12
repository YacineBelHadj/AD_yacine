import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from AD_vibration.models.stat_model import MahalanobisDistance
class MahalanobisDistanceTestCase(unittest.TestCase):
    def test_mahalanobis_distance(self):
        # Create some test data
        X, _ = make_classification(n_samples=100, n_features=10, random_state=42)

        # Create a MahalanobisDistance instance
        m = MahalanobisDistance()

        # Fit the model to the test data
        m.fit(X)

        # Compute the Mahalanobis distance for the test data
        distance = m.transform(X)
        print(distance.shape)

        # Assert that the shape of the distance is as expected
        self.assertEqual(distance.shape, (100,))

        # Assert that the distance is greater than zero for all samples
        self.assertTrue(np.all(distance > 0))

        # Compute the Mahalanobis distance for a DataFrame of test data
        df = pd.DataFrame(X)
        distance = m.transform(df)

        # Assert that the shape of the distance is as expected
        self.assertEqual(distance.shape, (100,))

        # Assert that the distance is greater than zero for all samples
        self.assertTrue(np.all(distance > 0))


if __name__ == '__main__':
    unittest.main()
