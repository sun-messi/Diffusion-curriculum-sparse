import unittest
import torch
from libs.curriculum import EmbeddingSparsityMask, validate_time_range

class CurriculumTests(unittest.TestCase):
    def test_mask_and_gradients(self):
        for ratio in (0., .15, .5, .9, 1.):
            mask = EmbeddingSparsityMask(16)
            mask.set_sparsity(ratio)
            x = torch.ones(2, 3, 16, requires_grad=True)
            y = mask(x)
            active = max(1, int(16 * (1-ratio)))
            self.assertTrue(torch.equal(y[..., :active], x[..., :active]))
            self.assertEqual(torch.count_nonzero(y[..., active:]).item(), 0)
            y.sum().backward()
            self.assertTrue(torch.all(x.grad[..., :active] == 1))
            self.assertEqual(torch.count_nonzero(x.grad[..., active:]).item(), 0)
            mask.set_inference_mode(True)
            self.assertTrue(torch.equal(mask(x), x))
    def test_validation(self):
        for ratio in (-.1, 1.1, float('nan')):
            with self.assertRaises(ValueError):
                EmbeddingSparsityMask(16).set_sparsity(ratio)
        with self.assertRaises(ValueError):
            EmbeddingSparsityMask(16)(torch.ones(2, 8))
        for bounds in ((-.1,1), (.5,.5), (0,2), (float('nan'),1)):
            with self.assertRaises(ValueError): validate_time_range(*bounds)
        validate_time_range(0., 1.)

if __name__ == '__main__': unittest.main()
