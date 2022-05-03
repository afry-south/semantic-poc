from smartsearch.docformer import BaseDocformer, distilroberta, mpnet_base, clip, longformer

import torch
import unittest

class TestBaseDocformer(unittest.TestCase):

    def test_clean_single(self):
        bd = BaseDocformer()
        text1 = '(cid:1) (cid:2) Furthermore, the Build abstraction lacks any further such symmetry.8CONCLUSIONSWe'
        text2 = 'calculation chainPrevious dependency graphAll dependency graphs, cacheAlgorithmTopological,'
        cleaned1 = bd.clean(text1)
        cleaned2 = bd.clean(text2)
        self.assertEqual(cleaned1, 'Furthermore, the Build abstraction lacks any further such symmetry.8 CONCLUSIONSWe')
        self.assertEqual(cleaned2, 'calculation chain Previous dependency graph All dependency graphs, cache Algorithm Topological,')

    def test_clean_batch(self):
        bd = BaseDocformer()
        texts = [
            r'delta record with key K and value Vwhere V (cid:60) S deleted , it adds V to S present',
            r'.dll, type !uext.he lp or!Pa th\w inext\uext.he lp.If you om it the ExtensionDLL, the d']

        cleaned = bd.clean(texts)
        self.assertEqual(cleaned, [
            'delta record with key K and value Vwhere V S deleted , it adds V to S present',
            '.dll, type !uext.he lp or! Pa th w inextxt.he lp. If you om it the Extension DLL, the d'])
    
class TestLongformer(unittest.TestCase):

    def test_encode(self):
        bd = longformer()
        embedding = bd('Long string '*2000)
        self.assertEqual(1, embedding.shape[0])
        self.assertEqual(768, embedding.shape[1])
        self.assertAlmostEqual(1.0, torch.linalg.vector_norm(embedding, dim=1, ord=2).item(), 4)

        embedding = bd(['test 1', 'test 2', 'test 3'])
        self.assertEqual(3, embedding.shape[0])
        self.assertEqual(768, embedding.shape[1])

class TestCLIP(unittest.TestCase):

    def test_encode(self):
        bd = clip()
        embedding = bd('Long string '*2000)
        self.assertEqual(1, embedding.shape[0])
        self.assertEqual(512, embedding.shape[1])
        self.assertAlmostEqual(1.0, torch.linalg.vector_norm(embedding, dim=1, ord=2).item(), 4)

        embedding = bd(['test 1', 'test 2', 'test 3'])
        self.assertEqual(3, embedding.shape[0])
        self.assertEqual(512, embedding.shape[1])

class TestMpnetBase(unittest.TestCase):

    def test_encode(self):
        bd = mpnet_base()
        embedding = bd('Long string '*2000)
        self.assertEqual(1, embedding.shape[0])
        self.assertEqual(768, embedding.shape[1])
        self.assertAlmostEqual(1.0, torch.linalg.vector_norm(embedding, dim=1, ord=2).item(), 4)

        embedding = bd(['test 1', 'test 2', 'test 3'])
        self.assertEqual(3, embedding.shape[0])
        self.assertEqual(768, embedding.shape[1])

class TestRoberta(unittest.TestCase):

    def test_encode(self):
        bd = distilroberta()
        embedding = bd('Long string '*2000)
        self.assertEqual(1, embedding.shape[0])
        self.assertEqual(768, embedding.shape[1])
        self.assertAlmostEqual(1.0, torch.linalg.vector_norm(embedding, dim=1, ord=2).item(), 4)

        embedding = bd(['test 1', 'test 2', 'test 3'])
        self.assertEqual(3, embedding.shape[0])
        self.assertEqual(768, embedding.shape[1])

if __name__ == '__main__':
    unittest.main()