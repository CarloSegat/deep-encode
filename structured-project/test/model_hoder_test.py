import unittest

from src.model.ModelHolder import ModelHolder
from src.StatefullPreprocessor import StatefulPreprocessor
from src.DataLoader import DataLoader
from src.Utils import TARGET_COL_NAME, get_rows_with_missing_values


class ModelHolderTest(unittest.TestCase):

    def setUp(self):
        self.load_data()
        self.preprocess()

        self.model_holder = ModelHolder()
        preprocessed_complete = self.preprocessor.transform(self.loader.complete)
        self.model_holder.fit_all(preprocessed_complete, TARGET_COL_NAME)

        self.all_predictions_for_complete = self.model_holder.predict_with_all(preprocessed_complete.drop([TARGET_COL_NAME], axis=1))


    def test_evaluation(self):
        all_evaluations = self.model_holder.evaluate_all()
        self.assertGreater(len(all_evaluations), 0)
        self.assertEqual(set(self.model_holder.models.keys()), set(all_evaluations.keys()))

    def test_prediction_are_not_empty(self):
        self.assertGreater(len(self.all_predictions_for_complete), 0)
        first_array_of_predicitons = self.all_predictions_for_complete[list(self.model_holder.models.keys())[0]]
        self.assertEqual(len(first_array_of_predicitons), len(self.loader.complete))

    def test_there_are_as_many_predicitons_as_datapoints(self):
        first_array_of_predicitons = self.all_predictions_for_complete[list(self.model_holder.models.keys())[0]]
        self.assertEqual(len(first_array_of_predicitons), len(self.loader.complete))


    def test_predictions_mean_is_close_to_true_value_mean(self):

        first_array_of_predicitons = self.all_predictions_for_complete[list(self.model_holder.models.keys())[0]]
        self.assertAlmostEqual(first_array_of_predicitons.mean(),
                               self.loader.complete[TARGET_COL_NAME].mean(),
                               delta=5)

    def preprocess(self):
        self.preprocessor = StatefulPreprocessor(augment=True)
        self.preprocessor.fit(self.loader.stateless_preprocessed)

    def load_data(self):
        self.loader = DataLoader()
        self.loader.load_data()

if __name__ == "__main__":
    unittest.main()
