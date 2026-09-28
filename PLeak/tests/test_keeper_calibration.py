import ast
import math
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]
ns = {'re': re, 'math': math}
tree = ast.parse((ROOT / 'PromptKeeper/promptkeeper/hypothesis_test.py').read_text())
body = [n for n in tree.body if isinstance(n, ast.FunctionDef)
        and n.name in ('parse_calibration_questions', 'validate_calibration_samples')]
exec(compile(ast.Module(body=body, type_ignores=[]), '<calibration>', 'exec'), ns)
parse = ns['parse_calibration_questions']
validate = ns['validate_calibration_samples']


class CalibrationTests(unittest.TestCase):
    def test_number_formats_and_no_final_newline(self):
        self.assertEqual(parse('1: First?\n2. Second?\n3) Third?\n**4.** Fourth?', 4),
                         ['First?', 'Second?', 'Third?', 'Fourth?'])

    def test_truncated_list_rejected(self):
        with self.assertRaisesRegex(ValueError, 'parsed 1'):
            parse('1: Only question?', 10)

    def test_prose_is_not_a_question(self):
        with self.assertRaises(ValueError):
            parse('Explanation\n1: One?\nMore explanation', 2)

    def test_one_sample_and_short_nonzero_variance_rejected(self):
        for values in ([-0.6], [-0.6, -0.4]):
            with self.assertRaises(ValueError):
                validate(values, 10)

    def test_constant_or_nonfinite_samples_rejected(self):
        for values in ([1.] * 10, [float('nan')] * 10, [float('inf')] * 10):
            with self.assertRaises(ValueError):
                validate(values, 10)

    def test_cached_samples_are_not_double_counted(self):
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'fit_distribution')
        appends = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Attribute) and n.func.attr == 'append'
                   and isinstance(n.func.value, ast.Name) and n.func.value.id == 'mll_list']
        self.assertEqual(len(appends), 1)

    def test_complete_varying_samples_accepted(self):
        validate([float(x) for x in range(10)], 10)


if __name__ == '__main__':
    unittest.main()
