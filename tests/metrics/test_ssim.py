import unittest
import torch

from torch import ones_like
from metrics.ssim import ssim


class TestSSIM(unittest.TestCase):
    a = torch.randn(4, 5, 10, 20) + .5
    ssim = ssim()

    @classmethod
    def bias_correction(cls, a: torch.Tensor):
        return 1 / torch.Tensor([a.size(d) for d in cls.ssim.dims]).prod()

    def test_shape(self):
        r = self.ssim(self.a, self.a)
        self.assertTrue(r.shape == self.a.shape[:2])
    
    def test_l_identical(self):
        r = self.ssim.l(self.a, self.a)
        self.assertTrue(r.allclose(torch.Tensor([1.])))

    def test_l_opposites(self):
        r = self.ssim.l(self.a, -self.a)
        self.assertTrue(r.lt(-.9).all())
    
    def test_l_zero(self):
        r = self.ssim.l(self.a.add(1), torch.zeros_like(self.a))
        self.assertTrue(r.allclose(torch.zeros(1), atol=1e-2))

    def test_c_identical(self):
        r = self.ssim.c(self.a, self.a)
        self.assertTrue(r.allclose(torch.Tensor([1])))
    
    def test_c_opposites(self):
        """
        Exact opposites should have identical standard deviations.
        """
        r = self.ssim.c(self.a, -self.a)
        self.assertTrue(r.allclose(torch.Tensor([-1.])))

    def test_c_zero_right(self):
        r = self.ssim.c(self.a, torch.zeros_like(self.a))
        self.assertTrue(r.allclose(torch.zeros(1), atol=1e-2))
    
    def test_c_zero_both(self):
        r = self.ssim.c(torch.zeros_like(self.a), torch.zeros_like(self.a))
        self.assertTrue(r.allclose(torch.ones(1)))

    def test_s_identical(self):
        r = self.ssim.s(self.a, self.a)
        self.assertTrue(r.gt(torch.Tensor([.99])).all())
        
    def test_s_zero_var(self):
        r = self.ssim.s(ones_like(self.a), ones_like(self.a))
        self.assertTrue(
            r.allclose(torch.Tensor([1.])))

    def test_s_opposites(self):
        r = self.ssim.s(self.a, -self.a)
        self.assertTrue(r.lt(torch.Tensor([-.99])).all())

    def test_s_zero(self):
        # The zero tensor should match perfectly to itself.
        r = self.ssim.s(torch.zeros_like(self.a), torch.zeros_like(self.a))
        self.assertTrue(r.allclose(torch.ones(1)))
    
    def test_s_right_zero(self):
        r = self.ssim.s(self.a, torch.zeros_like(self.a))
        self.assertTrue(r.allclose(torch.zeros(1)))

    def test_identical(self):
        r = self.ssim(self.a, self.a)
        self.assertTrue(r.gt(torch.Tensor([.995])).all())


    def test_opposites(self):
        r = self.ssim(self.a, -self.a)
        # Contast is identical; returns 1.
        self.assertTrue((r < 0.02).all())


    def test_scaled_opposites(self):
        r = self.ssim(self.a, -.1 * self.a)
        self.assertTrue(r.lt(-.2).all())

    def test_zero_right(self):
        """
        SSIM matching against the zero matrix should have minimal returns.
        l = 0; c = 0; s = 0;
        """
        r = self.ssim(self.a, torch.zeros_like(self.a))
        self.assertTrue(r.allclose(torch.zeros(1), atol=1e-2))

    def test_zero_both(self):
        """
        SSIM of two zero matrices should return the maximum value.
        """
        r = self.ssim(torch.zeros(5, 5, 10, 10), torch.zeros(5, 5, 10, 10))
        self.assertTrue(r.allclose(torch.ones(1), atol=1e-2))
