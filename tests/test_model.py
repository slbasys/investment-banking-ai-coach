"""
Test module for the Finance Bro AI Investment Banking Interview Coach model.

This module contains unit tests for the core model functionality, including:
- Basic model response testing
- Mode detection testing
- Edge case handling
- Company detection and financial data integration
"""

import unittest
import time
import os
import sys
from unittest.mock import patch

# Add the parent directory to sys.path to allow imports from the src module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import process_finance_query, determine_mode_by_keywords, detect_company_ticker
from src.utils import format_latex_content

class TestModeDetection(unittest.TestCase):
    """Tests for the mode detection functionality"""
    
    def test_experience_level_detection(self):
        """Test that experience levels are correctly detected from queries"""
        # Test junior mode
        experience_level, _ = determine_mode_by_keywords("I'm a beginner in finance")
        self.assertEqual(experience_level, "junior")
        
        # Test experienced mode
        experience_level, _ = determine_mode_by_keywords("As an experienced MBA candidate")
        self.assertEqual(experience_level, "experienced")
        
        # Test standard mode (default)
        experience_level, _ = determine_mode_by_keywords("How do you calculate WACC?")
        self.assertEqual(experience_level, "standard")
    
    def test_bro_mode_detection(self):
        """Test that bro mode is correctly detected from queries"""
        # Test bro mode on
        _, bro_mode = determine_mode_by_keywords("Hey bro, tell me about DCF")
        self.assertTrue(bro_mode)
        
        # Test bro mode off
        _, bro_mode = determine_mode_by_keywords("How do you calculate WACC?")
        self.assertFalse(bro_mode)
        
        # Test with finance bro phrase
        _, bro_mode = determine_mode_by_keywords("Help me crush this interview like a finance bro")
        self.assertTrue(bro_mode)


class TestCompanyDetection(unittest.TestCase):
    """Tests for company detection functionality"""
    
    def test_company_name_detection(self):
        """Test company detection from company names in queries"""
        self.assertEqual(detect_company_ticker("Tell me about Apple"), "AAPL")
        self.assertEqual(detect_company_ticker("What is Microsoft's business model?"), "MSFT")
        self.assertEqual(detect_company_ticker("Is Goldman Sachs a good company?"), "GS")
    
    def test_ticker_detection(self):
        """Test company detection from ticker symbols in queries"""
        self.assertEqual(detect_company_ticker("What about AAPL?"), "AAPL")
        self.assertEqual(detect_company_ticker("Compare MSFT and GOOGL"), "MSFT")
        self.assertEqual(detect_company_ticker("TSLA performance"), "TSLA")
    
    def test_no_company_detection(self):
        """Test that no company is detected when none is mentioned"""
        self.assertEqual(detect_company_ticker("What is WACC?"), "")
        self.assertEqual(detect_company_ticker("Tell me about investment banking"), "")


class TestModelResponses(unittest.TestCase):
    """Tests for model responses across different question types"""
    
    def setUp(self):
        """Set up test environment"""
        # Skip these tests if API keys aren't available
        if not os.environ.get("OPENAI_API_KEY"):
            self.skipTest("OpenAI API key not available")
    
    def test_technical_question_response(self):
        """Test model response to technical questions"""
        response = process_finance_query("What is WACC?", category="technical")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 100)  # Response should be substantial
        self.assertTrue("cost of capital" in response.lower())  # Should mention key concept
    
    @patch('src.model.detect_company_ticker')
    def test_company_specific_question(self, mock_detect):
        """Test model response to company-specific questions"""
        # Mock the company detection to return a specific ticker
        mock_detect.return_value = "AAPL"
        
        response = process_finance_query("What is Apple's business?")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 100)
        self.assertTrue("Apple" in response)  # Should mention the company
    
    def test_bro_mode_response(self):
        """Test that bro mode affects the response style"""
        standard_response = process_finance_query("What is DCF?")
        bro_response = process_finance_query("Bro, what is DCF?")
        
        # Bro response should contain casual language
        bro_indicators = ["bro", "crush", "street", "beast"]
        
        # Check if any bro indicators are in the bro response but not in standard
        bro_language_detected = any(
            term in bro_response.lower() and term not in standard_response.lower() 
            for term in bro_indicators
        )
        
        self.assertTrue(bro_language_detected)


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and error handling"""
    
    def setUp(self):
        """Set up test environment"""
        # Skip these tests if API keys aren't available
        if not os.environ.get("OPENAI_API_KEY"):
            self.skipTest("OpenAI API key not available")
    
    def test_empty_query(self):
        """Test handling of empty queries"""
        response = process_finance_query("")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)  # Should return some kind of response
    
    def test_very_short_query(self):
        """Test handling of very short queries"""
        response = process_finance_query("WACC?")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 100)  # Should still give a meaningful response
    
    def test_non_finance_question(self):
        """Test handling of non-finance questions"""
        response = process_finance_query("What's the weather like today?")
        self.assertIsInstance(response, str)
        self.assertTrue(
            "finance" in response.lower() or 
            "investment banking" in response.lower() or
            "interview" in response.lower()
        )  # Should redirect to finance topics
    
    def test_special_characters(self):
        """Test handling of special characters in query"""
        response = process_finance_query("What's the impact of COVID-19 on M&A? $$$")
        self.assertIsInstance(response, str)
        self.assertTrue("m&a" in response.lower())  # Should still address the core question


class TestPerformanceMetrics(unittest.TestCase):
    """Tests for performance metrics like response time"""
    
    def setUp(self):
        """Set up test environment"""
        # Skip these tests if API keys aren't available
        if not os.environ.get("OPENAI_API_KEY"):
            self.skipTest("OpenAI API key not available")
    
    def test_response_time(self):
        """Test that responses are generated within a reasonable time"""
        start_time = time.time()
        process_finance_query("What is enterprise value?")
        end_time = time.time()
        
        response_time = end_time - start_time
        self.assertLess(response_time, 60)  # Response should take less than 60 seconds
    
    def test_response_length(self):
        """Test that responses are of substantial length"""
        response = process_finance_query("What is enterprise value?")
        word_count = len(response.split())
        self.assertTrue(word_count > 50)  # Response should have at least 50 words


if __name__ == "__main__":
    unittest.main()
