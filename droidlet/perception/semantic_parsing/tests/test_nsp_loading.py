import os
import unittest

from droidlet.perception.semantic_parsing.nsp_transformer_model.query_model import NSPBertModel as Model

NLU_MODEL_DIR = os.path.join(
    os.path.dirname(__file__), "../../../../droidlet/artifacts/models/nlu/"
)
NLU_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "../../../../droidlet/artifacts/datasets/annotated_data/"
)

class TestNSPModel(unittest.TestCase):
    def setUp(self):
        self.nsp_model_dir = os.path.join(NLU_MODEL_DIR, "ttad_bert_updated")
        self.model = Model(model_dir=self.nsp_model_dir, data_dir=NLU_DATA_DIR)

    def test_model_parse(self):
        chat = 'come here'
        logical_form = self.model.parse(chat=chat)
        self.assertEqual(type(logical_form), dict)
        self.assertTrue("dialogue_type" in logical_form)
        chat = 'hello'
        logical_form = self.model.parse(chat=chat)
        self.assertEqual(type(logical_form), dict)
        self.assertTrue("dialogue_type" in logical_form)
        chat = 'dance'
        logical_form = self.model.parse(chat=chat)
        self.assertEqual(type(logical_form), dict)
        self.assertTrue("dialogue_type" in logical_form)

    def test_model_dir(self):
        # change the model directory and assert model doesn't load
        self.assertRaises(Exception, Model, self.nsp_model_dir + "wert", NLU_DATA_DIR)

if __name__ == '__main__':
    unittest.main()
