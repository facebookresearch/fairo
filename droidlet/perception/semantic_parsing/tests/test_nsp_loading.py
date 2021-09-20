import os
import unittest

from droidlet.perception.semantic_parsing.nsp_transformer_model.query_model import NSPBertModel as Model

NLU_MODEL_DIR = os.path.join(
    os.path.dirname(__file__), "../../../../agents/craftassist/models/semantic_parser/"
)
NLU_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "../../../../agents/craftassist/datasets/annotated_data/"
)

class TestNSPModel(unittest.TestCase):
    def setUp(self):
        nsp_data_dir = NLU_DATA_DIR
        nsp_model_dir = os.path.join(NLU_MODEL_DIR, "ttad_bert_updated")
        self.model = Model(model_dir=nsp_model_dir, data_dir=nsp_data_dir)

    def test_model_parse(self):
        chat = 'come here'
        logical_form = self.model.parse(chat=chat)
        self.assertEqual(type(logical_form), dict)
        chat = 'hello'
        logical_form = self.model.parse(chat=chat)
        self.assertEqual(type(logical_form), dict)
        chat = 'dance'
        logical_form = self.model.parse(chat=chat)
        self.assertEqual(type(logical_form), dict)

if __name__ == '__main__':
    unittest.main()
