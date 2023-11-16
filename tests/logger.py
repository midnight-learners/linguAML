import unittest
from pathlib import Path
from linguaml._logger import logger

class TestLogger(unittest.TestCase):
    
    def __init__(self, methodName: str = "runTest") -> None:
        
        super().__init__(methodName)
        
        # Reset the log file path
        logger.log_filepath = Path(__file__).parent.joinpath("test.log")
    
    def test_logger(self):
        
        logger.info("Hello, there")
        print(logger.log_filepath)

if __name__ == '__main__':
    unittest.main()