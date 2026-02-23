import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

file_path = Path(__file__)
top_dir = file_path.parent.parent

# Update python PATH so that we can load run.py from CTF_NaiveBaselines directly
sys.path.insert(0, str(top_dir))

from run import main as baseline_main

class TestBaseline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Called once before running all unit tests
        """
        pass

    @classmethod
    def tearDownClass(cls):
        """
        Called once after running all unit tests
        """
        pass

    def setUp(self):
        """
        Called once before each unit test
        """
        # Create a temporary directory for storing results
        self.temp_dir = tempfile.TemporaryDirectory(delete=False)

        # Set up patcher to use temporary directory when saving results
        self.patcher = patch('ctf4science.eval_module.top_dir', Path(self.temp_dir.name))
        self.patcher.start()

    def tearDown(self):
        """
        Called once after each unit test
        """
        self.patcher.stop()
        self.temp_dir.cleanup()

    def test_run_all_baseline_configs(self):
        """
        Test that CTF_NaiveBaselines runs with all current config files
        """
        # Get all relevant directories
        config_dir = top_dir / 'config'
        run_path = top_dir / 'run.py'
        self.assertTrue(config_dir.exists())
        self.assertTrue(config_dir.is_dir())
        self.assertTrue(run_path.exists())
        self.assertTrue(run_path.is_file())

        # Run for each config, make sure it runs without error
        for config_path in config_dir.iterdir():
            print("--------------------------------------------")
            print(f"{file_path.name}:")
            print("Running CTF_NaiveBaselines with config_path:", config_path)
            print("--------------------------------------------")
            baseline_main(config_path)
            break
        
    def test_run_all_hyperparameter_optimization_configs(self):
        """
        Test that CTF_NaiveBaselines runs with all current tuning config files
        """
        # Get all relevant directories
        config_dir = top_dir / 'tuning_config'
        run_path = top_dir / 'optimize_parameters.py'
        self.assertTrue(config_dir.exists())
        self.assertTrue(config_dir.is_dir())
        self.assertTrue(run_path.exists())
        self.assertTrue(run_path.is_file())

        # Run for each config, make sure it runs without error
        for config_path in config_dir.iterdir():
            cmd = f"python \"{run_path}\" --config-path \"{config_path}\""
            print("--------------------------------------------")
            print(f"{file_path.name}:")
            print(f"Running command \"{cmd}\"")
            print("--------------------------------------------")
            out = os.system(cmd)
            self.assertEqual(out, 0)