import unittest
import torch

from metrics.ssim import ssim


class TestSSIM(unittest.TestCase):
    a = torch.randn(4, 5, 6, 7)
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
        self.assertTrue((r < -.5).all())
    
    def test_l_zero(self):
        r = self.ssim.l(self.a, torch.zeros_like(self.a))
        self.assertTrue(r.allclose(torch.zeros(1), atol=1e-4))

    def test_c_identical(self):
        r = self.ssim.c(self.a, self.a)
        self.assertTrue(r.allclose(torch.Tensor([1])))
    
    def test_c_opposites(self):
        r = self.ssim.c(self.a, -self.a)
        # Exact opposites should have identical standard deviations.
        self.assertTrue(r.allclose(torch.Tensor([1.])))

    def test_c_zero(self):
        r = self.ssim.c(self.a, torch.zeros_like(self.a))
        self.assertTrue(r.allclose(torch.zeros(1)))

    def test_s_identical(self):
        r = self.ssim.s(self.a, self.a)
        self.assertTrue(
            r.allclose(torch.Tensor([1.]) - self.bias_correction(self.a)))
    
    def test_s_opposites(self):
        r = self.ssim.s(self.a, -self.a)
        self.assertTrue(
            r.allclose(- torch.ones(1) + self.bias_correction(self.a)))
        
    def test_s_zero(self):
        r = self.ssim.s(self.a, torch.zeros_like(self.a))
        self.assertTrue(r.allclose(torch.zeros(1)))
        
    def test_identical(self):
        r = self.ssim(self.a, self.a)
        self.assertTrue(
            r.allclose(torch.Tensor([3.]) - self.bias_correction(self.a)))

    def test_opposites(self):
        # This needs to be negative and close to -1;
        r = self.ssim(self.a, -self.a)
        self.assertTrue(
            r.allclose(-torch.ones(1) + self.bias_correction(self.a)))
        
    def test_scaled_opposites(self):
        # This needs to be negative and close to -1;
        r = self.ssim(self.a, -.1 * self.a)
        self.assertTrue(
            r.allclose(-torch.ones(1) + self.bias_correction(self.a)))

    def test_zero(self):
        r = self.ssim(self.a, torch.zeros_like(self.a))
        self.assertTrue(r.allclose(torch.zeros(1), atol=1e-4))
