"""
Tests for location identifier module
"""

import unittest
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock
import json

from photo_splitter.location_identifier import LocationIdentifier


class TestLocationIdentifier(unittest.TestCase):
    """Test cases for LocationIdentifier class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock the connection check to avoid actual API calls
        with patch.object(LocationIdentifier, '_check_connection'):
            self.identifier = LocationIdentifier()
    
    def test_identifier_initialization(self):
        """Test identifier initializes with correct default values"""
        with patch.object(LocationIdentifier, '_check_connection'):
            identifier = LocationIdentifier()
            self.assertEqual(identifier.ollama_url, "http://localhost:11434")
            self.assertEqual(identifier.model, "llava:latest")
    
    def test_identifier_custom_initialization(self):
        """Test identifier initializes with custom values"""
        with patch.object(LocationIdentifier, '_check_connection'):
            identifier = LocationIdentifier(
                ollama_url="http://custom:8080",
                model="custom-model"
            )
            self.assertEqual(identifier.ollama_url, "http://custom:8080")
            self.assertEqual(identifier.model, "custom-model")
    
    def test_encode_image(self):
        """Test image encoding to base64"""
        # Create a simple test image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[25:75, 25:75] = [255, 255, 255]
        
        # Encode the image
        encoded = self.identifier._encode_image(image)
        
        # Check that it's a non-empty string
        self.assertIsInstance(encoded, str)
        self.assertGreater(len(encoded), 0)
    
    def test_parse_response_with_location(self):
        """Test parsing LLM response with location"""
        response_text = """Location: Paris, France
Confidence: high
Description: I can see the Eiffel Tower in the background"""
        
        result = self.identifier._parse_response(response_text)
        
        self.assertEqual(result['location'], 'Paris, France')
        self.assertEqual(result['confidence'], 'high')
        self.assertIn('Eiffel Tower', result['description'])
    
    def test_parse_response_unknown_location(self):
        """Test parsing LLM response with unknown location"""
        response_text = """Location: Unknown
Confidence: N/A
Description: The image shows a street scene but no identifiable landmarks"""
        
        result = self.identifier._parse_response(response_text)
        
        self.assertIsNone(result['location'])
        self.assertIsNone(result['confidence'])
        self.assertIn('street scene', result['description'])
    
    def test_parse_response_partial_info(self):
        """Test parsing LLM response with partial information"""
        response_text = """Location: New York City, USA
Description: I can see tall buildings and yellow taxis"""
        
        result = self.identifier._parse_response(response_text)
        
        self.assertEqual(result['location'], 'New York City, USA')
        self.assertIsNone(result['confidence'])
        self.assertIn('tall buildings', result['description'])
    
    @patch('photo_splitter.location_identifier.requests.post')
    def test_identify_location_success(self, mock_post):
        """Test successful location identification"""
        # Mock successful API response
        mock_response = Mock()
        mock_response.json.return_value = {
            'response': """Location: London, UK
Confidence: medium
Description: I can see Big Ben in the background"""
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        # Create a test image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Identify location
        result = self.identifier.identify_location(image)
        
        # Verify the result
        self.assertEqual(result['location'], 'London, UK')
        self.assertEqual(result['confidence'], 'medium')
        self.assertIn('Big Ben', result['description'])
        
        # Verify API was called
        mock_post.assert_called_once()
    
    @patch('photo_splitter.location_identifier.requests.post')
    def test_identify_location_api_error(self, mock_post):
        """Test location identification with API error"""
        # Mock API error
        import requests
        mock_post.side_effect = requests.exceptions.RequestException("Connection error")
        
        # Create a test image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Identify location
        result = self.identifier.identify_location(image)
        
        # Should return error information
        self.assertIsNone(result['location'])
        self.assertIsNone(result['confidence'])
        self.assertIn('Error', result['description'])
    
    @patch('photo_splitter.location_identifier.requests.get')
    def test_check_connection_success(self, mock_get):
        """Test successful connection check"""
        # Mock successful connection
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        # Should not raise exception
        try:
            identifier = LocationIdentifier()
        except ConnectionError:
            self.fail("ConnectionError was raised unexpectedly")
    
    @patch('photo_splitter.location_identifier.requests.get')
    def test_check_connection_failure(self, mock_get):
        """Test connection check failure"""
        # Mock connection failure
        import requests
        mock_get.side_effect = requests.exceptions.ConnectionError("Cannot connect")
        
        # Should raise ConnectionError
        with self.assertRaises(ConnectionError):
            identifier = LocationIdentifier()


if __name__ == '__main__':
    unittest.main()
