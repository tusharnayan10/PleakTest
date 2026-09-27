import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[2]


class SimilarityCompatibilityTests(unittest.TestCase):
    def test_legacy_encoder_uses_all_pairs_cosine_and_same_averaging(self):
        source = (ROOT / 'prompt_obfuscation/src/output_similarity.py').read_text()
        tree = ast.parse(source)
        method = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'cosine_similarity')
        encode = Mock(side_effect=['pred_embeddings', 'ref_embeddings'])
        # An old encoder has neither model.similarity nor trust_remote_code.
        def constructor(name, device):
            self.assertEqual(name, 'all-mpnet-base-v2')
            self.assertEqual(device, 'cpu')
            return SimpleNamespace(encode=encode)
        matrix = [[1.0, 0.0], [0.0, 1.0]]
        cosine = Mock(return_value=SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: matrix)))
        def mean(value):
            self.assertEqual(value, matrix)
            return SimpleNamespace(item=lambda: 0.5)
        def nanmean(value):
            self.assertEqual(value, [0.5])
            return SimpleNamespace(item=lambda: 0.5)
        ns = {'SentenceTransformer': constructor, 'util': SimpleNamespace(cos_sim=cosine),
              'torch': SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
              'np': SimpleNamespace(nan=float('nan'), mean=mean, nanmean=nanmean),
              'get_gpu_utilization': lambda: 0, 'logger': Mock()}
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<cosine>', 'exec'), ns)
        self.assertEqual(ns['cosine_similarity']([['p1', 'p2']], [['r1', 'r2']]), (0.5, 0))
        cosine.assert_called_once_with('pred_embeddings', 'ref_embeddings')
        ns['logger'].warning.assert_not_called()
        imports = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == 'sentence_transformers']
        self.assertNotIn('SimilarityFunction', [alias.name for n in imports for alias in n.names])


if __name__ == '__main__':
    unittest.main()
