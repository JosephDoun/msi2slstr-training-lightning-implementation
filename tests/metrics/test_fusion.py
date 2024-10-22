import unittest

from metrics.fusion import msi2slstr_loss
from torch import randn, zeros_like, ones


class TestMSI2SLSTRLoss(unittest.TestCase):
    loss = msi2slstr_loss()
    a = randn(5, 13, 100, 100)
    b = randn(5, 12, 100, 100)
    c = randn(5, 12, 2, 2)
    d = randn(5, 6, 2, 2)
    e = randn(5, 6, 100, 100)
    
    def test_shape(self):
        r, _, _ = self.loss(self.a, self.b, self.c, self.d, self.e)
        self.assertTrue(r.shape == (5, 12))

    def test_zero(self):
        r, _, _ = self.loss(zeros_like(self.a), zeros_like(self.b),
                      zeros_like(self.c), zeros_like(self.d),
                      zeros_like(self.e))
        self.assertTrue(r.allclose(ones(1) * 4))