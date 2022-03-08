import unittest
import numpy as np
from frozen.transforms import pixelbert_transform

class TestPixelBERT(unittest.TestCase):
    def test_piexelbert(self):   
        print(pixelbert_transform(256))

if __name__ == "__main__":
    unittest.main()