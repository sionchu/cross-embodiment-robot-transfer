"""Episode-boundary and checkpoint adapter checks on real local artifacts."""
import json
from pathlib import Path
import sys
import unittest
import av
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/local'))
from act import ACTPolicy, CanonicalDataset, image_tensor


@unittest.skipUnless((ROOT/'datasets/ur10e_10/meta/info.json').exists(),'Generate canonical dataset first')
class ACTDataChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=CanonicalDataset(ROOT/'datasets/ur10e_10',16)

    def test_chunks_do_not_cross_episode_boundary(self):
        lengths=[json.loads(x)['length'] for x in (ROOT/'datasets/ur10e_10/meta/episodes.jsonl').read_text().splitlines()]
        end=0
        for n in lengths:
            end+=n
            sample=self.data[end-1]
            self.assertEqual(sample['action_is_pad'].tolist(),[False]+[True]*15)
            self.assertTrue(torch.equal(sample['action'],self.data.actions[end-1].repeat(16,1)))
            start=self.data[end-n]
            self.assertFalse(start['action_is_pad'].any())
        self.assertEqual(end,len(self.data))

    def test_state_normalization_is_reversible(self):
        original=np.stack(pd.read_parquet(ROOT/'datasets/ur10e_10/data/chunk-000/episode_000000.parquet')['observation.state'])
        s=self.data.statistics['observation.state']
        recovered=self.data.states[:len(original)].numpy()*np.maximum(s['std'],1e-6)+s['mean']
        np.testing.assert_allclose(recovered,original,atol=1e-7,rtol=0)

    def test_invalid_image_rejected(self):
        with self.assertRaisesRegex(ValueError,'INVALID_RGB'):
            image_tensor(np.zeros((480,640,3),dtype=np.float32))

    @unittest.skipUnless((ROOT/'checkpoints/act-10-001/model.safetensors').exists(),'Train actual checkpoint first')
    def test_checkpoint_predicts_finite_physical_chunk(self):
        frame=pd.read_parquet(ROOT/'datasets/ur10e_10/data/chunk-000/episode_000000.parquet')
        with av.open(str(ROOT/'datasets/ur10e_10/videos/chunk-000/observation.images.front/episode_000000.mp4')) as c:
            rgb=next(c.decode(video=0)).to_ndarray(format='rgb24')
        policy=ACTPolicy(ROOT/'checkpoints/act-10-001')
        chunk=policy.predict({'state':frame['observation.state'].iloc[0],'front':rgb})
        self.assertEqual(chunk.shape,(16,6))
        self.assertTrue(np.isfinite(chunk).all())
        policy.reset()
        np.testing.assert_allclose(policy.predict({'state':frame['observation.state'].iloc[0],'front':rgb}),chunk,atol=1e-7)


if __name__=='__main__':
    unittest.main()
