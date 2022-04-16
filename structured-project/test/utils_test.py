import unittest
from src.DataLoader import DataLoader
from src.Utils import TARGET_COL_NAME, get_categorical_columns, get_numerical_columns, calculate_rmse


class TestUtils(unittest.TestCase):

    def setUp(self):
        self.loader = DataLoader()
        self.loader.load_data()

    def test_get_categorical_and_get_numerical_on_raw_data(self):
        raw = self.loader.raw
        self._assert_columns_dont_share_elements(raw)

    def test_get_categorical_and_get_numerical_on_complete_data(self):
        complete = self.loader.complete
        self._assert_columns_dont_share_elements(complete)

    def test_get_categorical_and_get_numerical_on_missing_data(self):
        missing = self.loader.missing_others
        self._assert_columns_dont_share_elements(missing)

    def test_get_categorical_and_get_numerical_on_no_target_data(self):
        missing = self.loader.missing_target
        self._assert_columns_dont_share_elements(missing)

    def test_columns_are_the_same_between_complete_and_missing_data(self):
        complete = self.loader.complete
        missing = self.loader.missing_others
        categorical_columns_raw = get_categorical_columns(missing)
        numerical_columns_raw = get_numerical_columns(missing)
        categorical_columns_complete = get_categorical_columns(complete)
        numerical_columns_complete = get_numerical_columns(complete)

        self.assert_delta_is_empty(categorical_columns_complete, categorical_columns_raw, missing)
        self.assert_delta_is_empty(numerical_columns_complete, numerical_columns_raw, missing)

    def assert_delta_is_empty(self, categorical_columns_complete, categorical_columns_raw, raw):
        delta = set(categorical_columns_raw) - set(categorical_columns_complete)
        if len(delta) > 0:
            number_of_na_cells = len([i for i, x in enumerate(raw[delta].isna().values.flatten()) if x])
            self.assertTrue(raw[delta].isna().values.flatten().all() or number_of_na_cells > len(raw[delta]) * 0.9)
        return delta

    def _assert_columns_dont_share_elements(self, raw):
        categorical_columns = get_categorical_columns(raw)
        numerical_columns = get_numerical_columns(raw)
        self.assertTrue(len(raw.columns) == len(categorical_columns) + len(numerical_columns))
        self.assertEqual(set(categorical_columns) - set(numerical_columns), set(categorical_columns))
        self.assertEqual(set(numerical_columns) - set(categorical_columns), set(numerical_columns))

    def test_rmse_calculation(self):
        self.assertEqual(calculate_rmse([1], [1]), 0)
        self.assertAlmostEqual(calculate_rmse([10.5, 0.5, 5.5], [10.6, 0.6, 5.6]), 0.1)
        self.assertEqual(calculate_rmse([1000, 2000, 1000], [900, 1900, 900]), 100)




