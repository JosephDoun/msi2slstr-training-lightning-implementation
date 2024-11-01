import unittest

from metrics.fusion import msi2slstr_loss
from torch import randn, zeros_like, ones, ones_like


class TestMSI2SLSTRLoss(unittest.TestCase):
    loss = msi2slstr_loss()
    a = randn(5, 13, 100, 100)
    b = randn(5, 12, 100, 100)
    c = randn(5, 12, 2, 2)
    d = randn(5, 6, 2, 2)
    e = randn(5, 6, 100, 100)

    def test_shape(self):
        r, t = self.loss(self.a, self.b, self.c, self.d, self.e)
        self.assertTrue(r.shape == (5, 12))
        self.assertTrue(t.shape == (5, 6))

    def test_all_zero(self):
        """
        Zero matrices should match, returning the maximum value.
        """
        r, t = self.loss(zeros_like(self.a), zeros_like(self.b),
                         zeros_like(self.c), zeros_like(self.d),
                         zeros_like(self.e))
        self.assertTrue(r.allclose(ones(1)))
        self.assertTrue(t.allclose(ones(1)))

    def test_all_pos_zero_var(self):
        """
        ...
        """
        r, t = self.loss(ones_like(self.a), ones_like(self.b),
                         ones_like(self.c), ones_like(self.d),
                         ones_like(self.e))
        self.assertTrue(r.allclose(ones(1)))
        self.assertTrue(t.allclose(ones(1)))

    def test_all_pos(self):
        """
        ...
        """
        r, t = self.loss(ones_like(self.a), ones_like(self.b),
                         ones_like(self.c), ones_like(self.d),
                         ones_like(self.e))
        self.assertTrue(r.allclose(ones(1)))
        self.assertTrue(t.allclose(ones(1)))
