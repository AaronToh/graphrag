import unittest
from typing import List
from haystack import Document
from eval import calculate_mrr, calculate_sas

class TestCalculateMRR(unittest.TestCase):
    def test_perfect_match_rank_1(self):
        gt_docs = [[Document(content="relevant")]]
        ret_docs = [[Document(content="relevant"), Document(content="irrelevant")]]
        self.assertEqual(calculate_mrr(gt_docs, ret_docs), 1.0)

    def test_match_rank_3(self):
        gt_docs = [[Document(content="relevant")]]
        ret_docs = [[Document(content="irrelevant1"), Document(content="irrelevant2"), Document(content="relevant")]]
        self.assertEqual(calculate_mrr(gt_docs, ret_docs), 1.0 / 3)

    def test_no_match(self):
        gt_docs = [[Document(content="relevant")]]
        ret_docs = [[Document(content="irrelevant1"), Document(content="irrelevant2")]]
        self.assertEqual(calculate_mrr(gt_docs, ret_docs), 0.0)

    def test_multiple_queries(self):
        gt_docs = [
            [Document(content="rel1")],
            [Document(content="rel2")],
        ]
        ret_docs = [
            [Document(content="irrel"), Document(content="rel1")],
            [Document(content="rel2")],
        ]
        self.assertEqual(calculate_mrr(gt_docs, ret_docs), (0.5 + 1.0) / 2)

    def test_empty_ground_truth(self):
        gt_docs = [[]]
        ret_docs = [[Document(content="something")]]
        self.assertEqual(calculate_mrr(gt_docs, ret_docs), 0.0)

class TestCalculateSAS(unittest.TestCase):
    def test_identical_answers(self):
        predicted = ["The answer is yes."]
        ground_truth = ["The answer is yes."]
        score = calculate_sas(predicted, ground_truth)
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_similar_answers(self):
        predicted = ["Machine learning is a field of AI."]
        ground_truth = ["AI includes machine learning as a subfield."]
        score = calculate_sas(predicted, ground_truth)
        self.assertGreater(score, 0.5)

    def test_dissimilar_answers(self):
        predicted = ["The sky is blue."]
        ground_truth = ["Computers process data."]
        score = calculate_sas(predicted, ground_truth)
        self.assertLess(score, 0.3)

    def test_empty_strings(self):
        predicted = [""]
        ground_truth = ["Something"]
        score = calculate_sas(predicted, ground_truth)
        self.assertEqual(score, 0.0)

if __name__ == "__main__":
    unittest.main()