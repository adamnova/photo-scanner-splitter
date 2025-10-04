"""
Location identification module using Ollama LLM
"""

import base64
import json
import cv2
import numpy as np
from typing import Optional, Dict
import requests


class LocationIdentifier:
    """Identifies the location of photos using Ollama LLM"""
    
    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "qwen2.5-vl:32b"):
        """
        Initialize the location identifier
        
        Args:
            ollama_url: URL of the Ollama API server
            model: Name of the Ollama model to use (must support vision)
        """
        self.ollama_url = ollama_url
        self.model = model
        self._check_connection()
    
    def _check_connection(self):
        """Check if Ollama server is accessible"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise ConnectionError(
                f"Cannot connect to Ollama server at {self.ollama_url}. "
                f"Please ensure Ollama is running. Error: {e}"
            )
    
    def _encode_image(self, image: np.ndarray) -> str:
        """
        Encode image to base64 string
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Base64 encoded image string
        """
        # Encode image to JPEG format
        _, buffer = cv2.imencode('.jpg', image)
        # Convert to base64
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return image_base64
    
    def identify_location(self, image: np.ndarray) -> Dict[str, Optional[str]]:
        """
        Identify the location of a photo using Ollama LLM
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary with location information:
            {
                'location': Identified location or None if unknown,
                'confidence': Confidence level (high/medium/low) or None,
                'description': Additional description from the model
            }
        """
        # Encode image
        image_base64 = self._encode_image(image)
        
        # Prepare prompt for the LLM
        prompt = (
            "Analyze this photograph and identify where it was taken. "
            "Consider landmarks, architecture, natural features, signs, or any other visible clues. "
            "Provide the location in the following format:\n"
            "Location: [City, Country] or [Specific landmark/place]\n"
            "Confidence: [high/medium/low]\n"
            "Description: [Brief explanation of why you think this is the location]\n\n"
            "If you cannot identify the location, respond with:\n"
            "Location: Unknown\n"
            "Confidence: N/A\n"
            "Description: [What you can see in the image]"
        )
        
        # Call Ollama API
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "images": [image_base64],
                    "stream": False
                },
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            response_text = result.get('response', '')
            
            # Parse the response
            location_info = self._parse_response(response_text)
            return location_info
            
        except requests.exceptions.RequestException as e:
            return {
                'location': None,
                'confidence': None,
                'description': f"Error calling Ollama API: {e}"
            }
    
    def _parse_response(self, response_text: str) -> Dict[str, Optional[str]]:
        """
        Parse the LLM response to extract location information
        
        Args:
            response_text: Raw response from LLM
            
        Returns:
            Parsed location information
        """
        location = None
        confidence = None
        description = None
        
        lines = response_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line.lower().startswith('location:'):
                location = line.split(':', 1)[1].strip()
                if location.lower() == 'unknown':
                    location = None
            elif line.lower().startswith('confidence:'):
                confidence = line.split(':', 1)[1].strip()
                if confidence.lower() in ['n/a', 'na']:
                    confidence = None
            elif line.lower().startswith('description:'):
                description = line.split(':', 1)[1].strip()
        
        # If description is not separated by lines, try to extract it as remaining text
        if description is None and response_text:
            # Take everything after the last recognized field
            remaining = response_text
            for prefix in ['Location:', 'Confidence:', 'Description:']:
                if prefix in remaining:
                    parts = remaining.split(prefix)
                    if len(parts) > 1:
                        remaining = parts[-1]
            description = remaining.strip() if remaining.strip() else None
        
        return {
            'location': location,
            'confidence': confidence,
            'description': description
        }
