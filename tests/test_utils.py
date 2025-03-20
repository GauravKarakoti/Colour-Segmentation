# tests/test_utils.py
import pytest
import sys
import os

# Add the parent directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your modules here
# from your_module import your_function

def test_example():
    """Example test function."""
    assert True

# Example test for an image processing utility
# def test_image_processing():
#     """Test image processing utilities."""
#     from your_module import process_image
#     result = process_image(test_image_path)
#     assert result is not None
#     assert result.shape == (height, width, channels)