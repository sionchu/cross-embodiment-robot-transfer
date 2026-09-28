"""Contract checks use official types/helpers, never mock model outputs as evidence."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/local'))
from groot import check_checkpoint, decode_action, load_contract, observation_to_groot


@unittest.skipUnless((ROOT/'.external/Isaac-GR00T/.git').exists(),'Run pinned setup first')
class GR00TContractChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modalities,_=load_contract(ROOT/'.external/Isaac-GR00T')

    def test_missing_batch_and_wrong_horizon_rejected(self):
        for shape in [(16,6),(1,15,6),(1,16,7)]:
            with self.assertRaisesRegex(ValueError,'INVALID_GR00T_ACTION'):
                decode_action({'eef_delta':np.zeros(shape,dtype=np.float32)})

    def test_nonfinite_action_rejected(self):
        with self.assertRaisesRegex(ValueError,'INVALID_GR00T_ACTION'):
            decode_action({'eef_delta':np.full((1,16,6),np.nan,dtype=np.float32)})

    def test_no_second_relative_conversion(self):
        # Synthetic values test decoding only; these are not model predictions.
        action=np.arange(96,dtype=np.float32).reshape(1,16,6)*1e-5
        np.testing.assert_array_equal(decode_action({'eef_delta':action}),action[0])

    def test_invalid_observation_rejected(self):
        with self.assertRaisesRegex(ValueError,'INVALID_STATE'):
            observation_to_groot({'state':[0]*5,'front':np.zeros((480,640,3),dtype=np.uint8)},self.modalities)

    def test_old_model_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/'config.json').write_text(json.dumps({'model_type':'Gr00tN1d6'}))
            with self.assertRaisesRegex(ValueError,'NOT_N17_CHECKPOINT'):
                check_checkpoint(p,self.modalities)

    def test_base_model_without_target_embodiment_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/'config.json').write_text(json.dumps({'model_type':'Gr00tN1d7','model_name':'nvidia/Cosmos-Reason2-2B'}))
            (p/'processor_config.json').write_text(json.dumps({'processor_kwargs':{'modality_configs':{}}}))
            with self.assertRaisesRegex(ValueError,'TARGET_MODALITY_MISMATCH'):
                check_checkpoint(p,self.modalities)
