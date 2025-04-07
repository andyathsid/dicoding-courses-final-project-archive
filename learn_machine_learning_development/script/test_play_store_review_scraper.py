#!/usr/bin/env python3
"""
Test script for play_store_review_scraper.py

This script contains unit tests for the functions in play_store_review_scraper.py
"""

import unittest
from unittest.mock import patch, mock_open, MagicMock
import tempfile
import os
import sys
import csv
from google_play_scraper import Sort

# Import the module to test
from script.play_store_scraper import fetch_reviews, save_to_csv, main


class TestPlayStoreReviewScraper(unittest.TestCase):
    """Unit tests for the Play Store Review Scraper."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_reviews = [
            {
                'reviewId': '123',
                'userName': 'Test User',
                'content': 'Great app!',
                'score': 5,
                'thumbsUpCount': 10,
                'reviewCreatedVersion': '1.0',
                'at': '2023-01-01 12:00:00',
                'replyContent': None,
                'repliedAt': None
            },
            {
                'reviewId': '456',
                'userName': 'Another User',
                'content': 'Could be better',
                'score': 3,
                'thumbsUpCount': 2,
                'reviewCreatedVersion': '1.0',
                'at': '2023-01-02 13:00:00',
                'replyContent': 'Thanks for your feedback',
                'repliedAt': '2023-01-03 09:00:00'
            }
        ]
        self.app_id = 'com.example.app'
    
    @patch('play_store_review_scraper.reviews_all')
    def test_fetch_reviews_success(self, mock_reviews_all):
        """Test that fetch_reviews successfully retrieves reviews."""
        mock_reviews_all.return_value = self.sample_reviews
        
        result = fetch_reviews(self.app_id)
        
        mock_reviews_all.assert_called_once_with(
            app_id=self.app_id,
            lang='en',
            country='us',
            sort=Sort.NEWEST
        )
        self.assertEqual(result, self.sample_reviews)
        self.assertEqual(len(result), 2)
        
    @patch('play_store_review_scraper.reviews_all')
    def test_fetch_reviews_with_params(self, mock_reviews_all):
        """Test fetch_reviews with custom parameters."""
        mock_reviews_all.return_value = self.sample_reviews
        
        result = fetch_reviews(
            self.app_id,
            language='fr',
            country='fr',
            sort_method=Sort.RATING
        )
        
        mock_reviews_all.assert_called_once_with(
            app_id=self.app_id,
            lang='fr',
            country='fr',
            sort=Sort.RATING
        )
        self.assertEqual(result, self.sample_reviews)
    
    @patch('play_store_review_scraper.reviews_all')
    def test_fetch_reviews_error(self, mock_reviews_all):
        """Test that fetch_reviews handles errors properly."""
        mock_reviews_all.side_effect = Exception("API Error")
        
        with self.assertRaises(Exception):
            fetch_reviews(self.app_id)
    
    def test_save_to_csv_success(self):
        """Test that save_to_csv correctly writes reviews to a CSV file."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            filename = tmp_file.name
            
        try:
            save_to_csv(self.sample_reviews, filename)
            
            with open(filename, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                self.assertEqual(len(rows), 2)
                self.assertEqual(rows[0]['reviewId'], '123')
                self.assertEqual(rows[0]['userName'], 'Test User')
                self.assertEqual(rows[1]['reviewId'], '456')
                self.assertEqual(rows[1]['content'], 'Could be better')
        finally:
            if os.path.exists(filename):
                os.unlink(filename)
    
    def test_save_to_csv_empty_reviews(self):
        """Test that save_to_csv handles empty review lists."""
        save_to_csv([], 'nonexistent.csv')
        self.assertFalse(os.path.exists('nonexistent.csv'))
    
    @patch('builtins.open', new_callable=mock_open)
    def test_save_to_csv_error(self, mock_file):
        """Test that save_to_csv handles file operation errors."""
        mock_file.side_effect = IOError("File error")
        
        with self.assertRaises(IOError):
            save_to_csv(self.sample_reviews, 'test.csv')
    
    @patch('play_store_review_scraper.fetch_reviews')
    @patch('play_store_review_scraper.save_to_csv')
    @patch('argparse.ArgumentParser.parse_args')
    def test_main_success(self, mock_parse_args, mock_save_to_csv, mock_fetch_reviews):
        """Test that the main function works properly."""
        mock_args = MagicMock()
        mock_args.app_id = self.app_id
        mock_args.output = 'output.csv'
        mock_args.language = 'en'
        mock_args.country = 'us'
        mock_args.sort = 'newest'
        mock_parse_args.return_value = mock_args
        
        mock_fetch_reviews.return_value = self.sample_reviews
        
        main()
        
        mock_fetch_reviews.assert_called_once_with(
            self.app_id,
            language='en',
            country='us',
            sort_method=Sort.NEWEST
        )
        mock_save_to_csv.assert_called_once_with(self.sample_reviews, 'output.csv')
    
    @patch('play_store_review_scraper.fetch_reviews')
    @patch('argparse.ArgumentParser.parse_args')
    @patch('sys.exit')
    def test_main_error(self, mock_exit, mock_parse_args, mock_fetch_reviews):
        """Test that the main function handles errors properly."""
        mock_args = MagicMock()
        mock_args.app_id = self.app_id
        mock_args.output = 'output.csv'
        mock_args.language = 'en'
        mock_args.country = 'us'
        mock_args.sort = 'newest'
        mock_parse_args.return_value = mock_args
        
        mock_fetch_reviews.side_effect = Exception("Test error")
        
        main()
        
        mock_exit.assert_called_once_with(1)


if __name__ == '__main__':
    unittest.main()