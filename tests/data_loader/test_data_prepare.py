import unittest
import numpy as np
from AD_vibration.data_loader.data_prepare import DataPrepare

class DataPrepareTestCase(unittest.TestCase):
    def test_prepare_data(self):
        # Create a DataPrepare instance with the appropriate parameters
        data_prepare = DataPrepare(12, 1000, True)

        # Create some test input data
        input_data = np.random.rand(3000, 12, 370)

        # Call the prepare_data method
        X_train, y_train, X, Y, data = data_prepare.prepare_data(input_data, return_data=True)

        # Assert that the shapes of the output data are as expected
        self.assertEqual(X_train.shape, (12000, 370))
        self.assertEqual(y_train.shape, (12000, 12))
        self.assertEqual(X.shape, (36000, 370))
        self.assertEqual(Y.shape, (36000,))
        self.assertEqual(data.shape, (3000, 12, 370))

        # Assert that the minimum and maximum values of the input data were properly scaled
        self.assertTrue(np.isclose(np.min(data), 0.0))
        self.assertTrue(np.isclose(np.max(data), 1.0))

if __name__ == '__main__':
    unittest.main()