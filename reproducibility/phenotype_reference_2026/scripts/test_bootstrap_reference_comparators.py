"""Boundary controls for weighted tied-score metrics; no clinical observations generated."""
import unittest
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score

from bootstrap_reference_comparators import score_metrics,cm_metrics


class WeightedMetricBoundaryTests(unittest.TestCase):
    def test_ties_and_unobserved_leading_groups(self):
        scores=np.array([.1,.1,.3,.7,.7]);y=np.array([0,1,0,1,0]);project=np.array([0,0,1,2,2])
        _,inverse=np.unique(scores,return_inverse=True)
        total=np.zeros((3,3));positive=np.zeros((3,3))
        np.add.at(total,(project,inverse),1);np.add.at(positive,(project,inverse),y)
        weights=np.array([[0,2,1],[2,0,1],[1,1,1]])
        auc,ap=score_metrics(weights,positive,total,batch_size=1)
        for i,w in enumerate(weights):
            self.assertAlmostEqual(auc[i],roc_auc_score(y,scores,sample_weight=w[project]),places=12)
            self.assertAlmostEqual(ap[i],average_precision_score(y,scores,sample_weight=w[project]),places=12)

    def test_empty_resistant_reference_marks_undefined(self):
        m=cm_metrics(np.array([[0,0,3,2]]))
        self.assertTrue(np.isnan(m['balanced_accuracy'][0]));self.assertTrue(np.isnan(m['mcc'][0]))
        self.assertTrue(np.isnan(m['false_susceptible_rate'][0]));self.assertAlmostEqual(m['false_resistant_rate'][0],.4)

    def test_constant_score_preserves_auc_and_ap(self):
        auc,ap=score_metrics(np.ones((1,1)),np.array([[3.]]),np.array([[5.]]))
        self.assertAlmostEqual(auc[0],.5);self.assertAlmostEqual(ap[0],.6)

    def test_no_positive_labels_ap_is_undefined_by_protocol(self):
        auc,ap=score_metrics(np.ones((1,1)),np.array([[0.,0.]]),np.array([[3.,2.]]))
        self.assertTrue(np.isnan(auc[0]));self.assertTrue(np.isnan(ap[0]))


if __name__=='__main__':unittest.main()
