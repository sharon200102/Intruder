import unittest
from pathlib import Path
from intruder.detections import VideoDetections
from intruder.evaluation.evaluate import compute_video_f1_score, VideoLevelMetrics
from . import GT_FILE_NAME, PRED_FILE_NAME,DATA_FOLDER_NAME

data_dir = Path(__file__).parent/DATA_FOLDER_NAME
class TestEvaluation(unittest.TestCase):
    def test_standard_evaluation_scenario(self):
        gt_detections = VideoDetections.from_mot_file(data_dir/GT_FILE_NAME)
        pred_detections = VideoDetections.from_mot_file(data_dir/PRED_FILE_NAME)
        evaluation_results = compute_video_f1_score(gt_detections,pred_detections)
        self.assertTrue(isinstance(evaluation_results,VideoLevelMetrics))
        self.assertGreater(evaluation_results.recall,0)
        self.assertGreater(evaluation_results.precision,0)
        self.assertGreater(evaluation_results.f1,0)
