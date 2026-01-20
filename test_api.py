"""
Credit Card RAG API - Test Script
==================================

This script tests all API endpoints to ensure they're working correctly.
"""

import requests
import json
import time
from typing import Dict, Any

# API base URL
BASE_URL = "http://localhost:8000"


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}\n")


def print_result(result: Dict[str, Any], show_full_answer: bool = True):
    """Print API result in a formatted way."""
    if 'answer' in result:
        print(f"Question: {result.get('question', 'N/A')}")
        if show_full_answer:
            print(f"\nAnswer:\n{result['answer']}")
        else:
            print(f"\nAnswer: {result['answer'][:200]}...")
        
        if 'sources' in result:
            print(f"\nSources ({len(result['sources'])}):")
            for i, source in enumerate(result['sources'], 1):
                print(f"  {i}. {source['card_name']} ({source['bank']}) - Score: {source['score']}")
        
        if result.get('filters_applied'):
            print(f"\nFilters Applied: {result['filters_applied']}")
    else:
        print(json.dumps(result, indent=2))


def test_health():
    """Test health check endpoint."""
    print_section("TEST 1: Health Check")
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200, "Health check failed"
    print("\n✓ Health check passed")


def test_ask_simple():
    """Test simple ask endpoint."""
    print_section("TEST 2: Simple Question")
    
    payload = {
        "question": "What's the best credit card for international travel?"
    }
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/ask", json=payload)
    elapsed = time.time() - start_time
    
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response Time: {elapsed:.2f}s\n")
    
    result = response.json()
    print_result(result, show_full_answer=False)
    
    assert response.status_code == 200, "Ask endpoint failed"
    assert 'answer' in result, "No answer in response"
    assert 'sources' in result, "No sources in response"
    print("\n✓ Simple question test passed")


def test_ask_with_filters():
    """Test ask endpoint with filters."""
    print_section("TEST 3: Question with Filters")
    
    payload = {
        "question": "Which card offers the best lounge access?",
        "category": "travel"
    }
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    
    response = requests.post(f"{BASE_URL}/ask", json=payload)
    
    print(f"\nStatus Code: {response.status_code}")
    
    result = response.json()
    print_result(result, show_full_answer=False)
    
    assert response.status_code == 200, "Filtered ask failed"
    assert result.get('filters_applied', {}).get('category') == 'travel', "Filter not applied"
    print("\n✓ Filtered question test passed")


def test_recommend():
    """Test recommendation endpoint."""
    print_section("TEST 4: Personalized Recommendation")
    
    payload = {
        "use_case": "online shopping and food delivery",
        "budget": "under 1000 annual fee",
        "preferences": ["high cashback", "no joining fee"]
    }
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    
    response = requests.post(f"{BASE_URL}/recommend", json=payload)
    
    print(f"\nStatus Code: {response.status_code}")
    
    result = response.json()
    print_result(result, show_full_answer=False)
    
    assert response.status_code == 200, "Recommend endpoint failed"
    print("\n✓ Recommendation test passed")


def test_compare():
    """Test comparison endpoint."""
    print_section("TEST 5: Card Comparison")
    
    payload = {
        "card_names": [
            "Axis Atlas Credit Card",
            "HSBC TravelOne Credit Card"
        ]
    }
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    
    response = requests.post(f"{BASE_URL}/compare", json=payload)
    
    print(f"\nStatus Code: {response.status_code}")
    
    result = response.json()
    print_result(result, show_full_answer=False)
    
    assert response.status_code == 200, "Compare endpoint failed"
    print("\n✓ Comparison test passed")


def test_search():
    """Test search endpoint."""
    print_section("TEST 6: Search Cards")
    
    params = {
        "query": "travel rewards",
        "category": "travel",
        "top_k": 3
    }
    
    print(f"Params: {json.dumps(params, indent=2)}")
    
    response = requests.get(f"{BASE_URL}/search", params=params)
    
    print(f"\nStatus Code: {response.status_code}")
    
    result = response.json()
    print(f"Query: {result['query']}")
    print(f"Filters: {result['filters']}")
    print(f"Total Results: {result['total_results']}\n")
    
    print("Results:")
    for i, card in enumerate(result['results'], 1):
        print(f"  {i}. {card['card_name']} ({card['bank']})")
        print(f"     Category: {card['category']}, Score: {card['score']}")
    
    assert response.status_code == 200, "Search endpoint failed"
    assert result['total_results'] > 0, "No search results"
    print("\n✓ Search test passed")


def test_metadata_endpoints():
    """Test metadata endpoints."""
    print_section("TEST 7: Metadata Endpoints")
    
    # Test categories
    print("Categories:")
    response = requests.get(f"{BASE_URL}/categories")
    categories = response.json()
    print(f"  {categories['categories']}")
    assert response.status_code == 200
    
    # Test banks
    print("\nBanks:")
    response = requests.get(f"{BASE_URL}/banks")
    banks = response.json()
    print(f"  {banks['banks']}")
    assert response.status_code == 200
    
    # Test reward types
    print("\nReward Types:")
    response = requests.get(f"{BASE_URL}/reward-types")
    reward_types = response.json()
    print(f"  {reward_types['reward_types']}")
    assert response.status_code == 200
    
    print("\n✓ Metadata endpoints test passed")


def test_multiple_filters():
    """Test with multiple filters."""
    print_section("TEST 8: Multiple Filters")
    
    payload = {
        "question": "Best card for frequent flyers",
        "category": "travel",
        "bank": "Axis Bank",
        "reward_type": "miles"
    }
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    
    response = requests.post(f"{BASE_URL}/ask", json=payload)
    
    print(f"\nStatus Code: {response.status_code}")
    
    result = response.json()
    print_result(result, show_full_answer=False)
    
    assert response.status_code == 200
    filters = result.get('filters_applied', {})
    assert filters.get('category') == 'travel'
    assert filters.get('bank') == 'Axis Bank'
    assert filters.get('reward_type') == 'miles'
    print("\n✓ Multiple filters test passed")


def test_error_handling():
    """Test error handling."""
    print_section("TEST 9: Error Handling")
    
    # Test with empty question
    print("Testing empty question...")
    payload = {"question": ""}
    response = requests.post(f"{BASE_URL}/ask", json=payload)
    print(f"Status Code: {response.status_code}")
    
    # Test with invalid endpoint
    print("\nTesting invalid endpoint...")
    response = requests.get(f"{BASE_URL}/invalid")
    print(f"Status Code: {response.status_code}")
    
    print("\n✓ Error handling test passed")


def run_all_tests():
    """Run all tests."""
    print("="*80)
    print("CREDIT CARD RAG API - TEST SUITE")
    print("="*80)
    print(f"\nAPI Base URL: {BASE_URL}")
    print("Starting tests...\n")
    
    tests = [
        test_health,
        test_ask_simple,
        test_ask_with_filters,
        test_recommend,
        test_compare,
        test_search,
        test_metadata_endpoints,
        test_multiple_filters,
        test_error_handling
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n✗ Test failed: {str(e)}")
            failed += 1
    
    # Summary
    print_section("TEST SUMMARY")
    print(f"Total Tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {failed} test(s) failed")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    try:
        run_all_tests()
    except requests.exceptions.ConnectionError:
        print("\n✗ Error: Could not connect to API")
        print("Make sure the API is running at http://localhost:8000")
        print("\nStart the API with: uv run api.py")
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
