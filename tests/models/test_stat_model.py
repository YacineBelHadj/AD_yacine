import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from AD_vibration.models.anomaly_index import MahalanobisDistance
class MahalanobisDistanceTestCase(unittest.TestCase):
    def test_mahalanobis_distance1(self):
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

    def test_mahalanobis_distance2(self):
        data = {'score': [91, 93, 72, 87, 86, 73, 68, 87, 78, 99, 95, 76, 84, 96, 76, 80, 83, 84, 73, 74],
                'hours': [16, 6, 3, 1, 2, 3, 2, 5, 2, 5, 2, 3, 4, 3, 3, 3, 4, 3, 4, 4],
                'prep': [3, 4, 0, 3, 4, 0, 1, 2, 1, 2, 3, 3, 3, 2, 2, 2, 3, 3, 2, 2],
                'grade': [70, 88, 80, 83, 88, 84, 78, 94, 90, 93, 89, 82, 95, 94, 81, 93, 93, 90, 89, 89]
                }

        df = pd.DataFrame(data,columns=['score', 'hours', 'prep','grade'])
        X = df[['score', 'hours', 'prep','grade']]
        m = MahalanobisDistance()
        m.fit(X)
        df['mahalanobis'] = m.transform(df[['score', 'hours', 'prep', 'grade']])

        true_mahalanobis = np.array([16.501963, 2.639286, 4.850797,
        5.201261, 3.828734, 4.090563, 
        4.283630, 2.419874, 1.651958, 
        5.657825, 3.965877, 2.935018, 
        2.810211, 4.368295, 1.561017, 
        1.459507, 2.024575, 0.750254, 
        2.735129, 2.264227])
        self.assertTrue(np.allclose(df['mahalanobis'].values,true_mahalanobis,rtol=1e-6, atol=1e-5))



if __name__ == '__main__':
    unittest.main()
