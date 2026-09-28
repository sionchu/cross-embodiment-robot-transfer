"""Corruption checks against the generated canonical dataset, when present."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('dataset_tools',ROOT/'scripts/local/dataset.py')
data=importlib.util.module_from_spec(spec); spec.loader.exec_module(data)


@unittest.skipUnless((ROOT/'datasets/ur10e_10/meta/info.json').exists(), 'Generate the real 10-episode dataset first')
class DatasetCorruptionChecks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)/'dataset'
        shutil.copytree(ROOT/'datasets/ur10e_10',self.root)
        self.parquet=self.root/'data/chunk-000/episode_000000.parquet'

    def tearDown(self):
        self.tmp.cleanup()

    def reject_rows(self, mutate, code):
        frame=pd.read_parquet(self.parquet)
        mutate(frame)
        frame.to_parquet(self.parquet,index=False)
        with self.assertRaisesRegex(ValueError,code):
            data.validate(self.root)

    def test_missing_column(self):
        self.reject_rows(lambda f:f.drop(columns=['action'],inplace=True),'MISSING_COLUMNS')

    def test_wrong_dimension(self):
        self.reject_rows(lambda f:f.__setitem__('action',[x[:5] for x in f.action]),'DIMENSION_MISMATCH')

    def test_nan_state(self):
        self.reject_rows(lambda f:f.at.__setitem__((0,'observation.state'),np.full(6,np.nan,dtype=np.float32)),'NONFINITE_DATA')

    def test_infinite_action(self):
        self.reject_rows(lambda f:f.at.__setitem__((0,'action'),np.full(6,np.inf,dtype=np.float32)),'NONFINITE_DATA')

    def test_duplicate_timestamp(self):
        self.reject_rows(lambda f:f.at.__setitem__((1,'timestamp'),0.),'INVALID_TIMESTAMPS')

    def test_episode_discontinuity(self):
        self.reject_rows(lambda f:f.at.__setitem__((1,'frame_index'),7),'EPISODE_DISCONTINUITY')

    def test_action_out_of_range(self):
        self.reject_rows(lambda f:f.at.__setitem__((0,'action'),np.ones(6,dtype=np.float32)),'ACTION_RANGE_VIOLATION')

    def test_missing_modality(self):
        path=self.root/'meta/modality.json'; value=json.loads(path.read_text()); del value['video']
        data.write_json(path,value)
        with self.assertRaisesRegex(ValueError,'MODALITY_MISMATCH'):
            data.validate(self.root)

    def test_train_eval_overlap(self):
        path=self.root/'meta/provenance.json'; value=json.loads(path.read_text())
        value['splits']['held_out'][0]=value['splits']['train'][0]; data.write_json(path,value)
        with self.assertRaisesRegex(ValueError,'TRAIN_EVAL_OVERLAP'):
            data.validate(self.root)

    def test_short_video(self):
        video=self.root/'videos/chunk-000/observation.images.front/episode_000000.mp4'
        short=self.root/'short.mp4'
        subprocess.run(['ffmpeg','-v','error','-i',str(video),'-frames:v','1','-c:v','libx264',str(short)],check=True)
        shutil.copyfile(short,video)
        with self.assertRaisesRegex(ValueError,'VIDEO_STATE_LENGTH_MISMATCH'):
            data.validate(self.root)


if __name__=='__main__':
    unittest.main()
