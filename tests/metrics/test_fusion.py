import unittest

from metrics.fusion import msi2slstr_loss
from torch import randn


class TestMSI2SLSTRLoss(unittest.TestCase):
    loss = msi2slstr_loss()
    a = randn(5, 13, 100, 100)
    b = randn(5, 12, 100, 100)
    c = randn(5, 12, 2, 2)
    d = randn(5, 6, 2, 2)
    e = randn(5, 6, 100, 100)
    
    def test_run(self):
        r = self.loss(self.a, self.b, self.c, self.d, self.e)
        print(r)
        self.assertTrue(r.shape == (5, 12))
