"""Geometry/termination checks; real PhysX and rendering are separate gates."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('insertion', Path(__file__).parents[1]/'scripts/local/insertion.py')
task = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task)


class InsertionChecks(unittest.TestCase):
    def test_success_requires_depth_alignment_and_no_collision(self):
        good = np.array([0., 0., .03, 0., 0., 0.])
        self.assertEqual(task.terminal(good, 0., 50), 'SUCCESS')
        for index,value in [(0,.002),(1,.002),(2,-.01),(2,.06),(3,.02),(4,.02),(5,.02)]:
            bad=good.copy()
            bad[index]=value
            self.assertNotEqual(task.terminal(bad, 0., 50), 'SUCCESS')
        self.assertEqual(task.terminal(good, 2., 50), 'COLLISION')

    def test_timeout_and_invalid_state(self):
        self.assertEqual(task.terminal(np.zeros(6),0.,task.CONFIG['max_steps']), 'TIMEOUT')
        self.assertEqual(task.terminal([float('nan')]*6,0.,1), 'NONFINITE_STATE')
        self.assertEqual(task.terminal(np.zeros(5),0.,1), 'NONFINITE_STATE')

    def test_expert_does_not_insert_while_misaligned(self):
        action,phase=task.expert(np.array([.015,0.,-.035,0.,0.,.1]))
        self.assertEqual(phase,'APPROACH_ALIGN')
        self.assertEqual(action[2],0.)
        self.assertLess(action[0],0.)
        self.assertLess(action[5],0.)
        action,phase=task.expert(np.array([0.,0.,-.035,0.,0.,0.]))
        self.assertEqual(phase,'INSERT')
        self.assertGreater(action[2],0.)


if __name__=='__main__':
    unittest.main()
