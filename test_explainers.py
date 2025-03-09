#!/usr/bin/env python3
"""
Test script for the XAI service with various explanation methods.
This script demonstrates how to call the API with different loan scenarios
and explanation methods.
"""
import json
import requests
import time
from typing import Dict, Any, List, Optional

# API endpoint - adjust as needed for your environment
API_URL = "http://127.0.0.1:5000"
AUTH_HEADER = {"X-API-Key": "XAI-dev-key-2023"}  # Development API key


def call_explain_api(loan_data, methods=None):
    """Call the explain API with the given loan data and methods."""
    # Add explanation methods if provided
    if methods:
        loan_data["explanation_methods"] = methods
    
    # Make the API request
    print(f"Calling API with data: {json.dumps(loan_data, indent=2)}")
    print("-" * 80)
    
    try:
        response = requests.post(
            f"{API_URL}/explain",
            json=loan_data,
            headers=AUTH_HEADER,
            timeout=10
        )
        
        # Handle response
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"Exception calling API: {str(e)}")
        return None


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)


def print_explanation_summary(explanation):
    """Print a summary of the explanation."""
    if not explanation:
        print("No explanation received")
        return
    
    # Print basic info
    request_id = explanation.get("request_id", "unknown")
    prediction = explanation.get("prediction", {})
    decision = prediction.get("decision", "unknown")
    score = prediction.get("score", 0)
    
    print(f"Request ID: {request_id}")
    print(f"Decision: {decision} (score: {score:.2f})")
    
    # Print confidence
    confidence = explanation.get("confidence", {})
    overall_score = confidence.get("overall_score", 0)
    print(f"Confidence: {overall_score:.2f}")
    
    # Print explanations
    print("\nExplanations:")
    for expl in explanation.get("explanations", []):
        method = expl.get("method", "unknown")
        model_type = expl.get("model_type", "unknown")
        print(f"\n- Method: {method} (model: {model_type})")
        
        # Print feature importance if available
        if expl.get("feature_importance"):
            print("  Feature Importance:")
            for feature, importance in expl.get("feature_importance", {}).items():
                print(f"  - {feature}: {importance:.3f}")
        
        # Print decision rules if available
        if expl.get("decision_rules"):
            print("  Decision Rules:")
            for rule in expl.get("decision_rules", []):
                print(f"  - {rule}")
        
        # Print counterfactuals if available
        if expl.get("counterfactuals"):
            print("  Counterfactuals:")
            for cf in expl.get("counterfactuals", [])[:2]:  # Show at most 2
                print(f"  - {cf.get('explanation', '')}")
                
        # Print feature interactions if available
        if expl.get("feature_interactions"):
            print("  Feature Interactions:")
            interactions = expl.get("feature_interactions", {})
            # Just show a sample of interactions
            for feature, inters in list(interactions.items())[:2]:
                print(f"  - {feature} interacts with:")
                for other, value in list(inters.items())[:3]:
                    print(f"    - {other}: {value:.2f}")
        
        # Print visualizations if available
        if expl.get("visualizations"):
            print("  Visualizations:")
            for viz in expl.get("visualizations", []):
                print(f"  - {viz.get('type')}: {viz.get('title')}")
    
    # Print computation time
    comp_time = explanation.get("computation_time", {})
    total_time = comp_time.get("total", 0)
    print(f"\nTotal computation time: {total_time:.2f} seconds")


def main():
    """Main function to test the XAI service."""
    print_header("XAI Service Test")
    
    # Test Case 1: Basic loan with feature importance
    print_header("Test Case 1: Basic Loan with Feature Importance")
    loan_data_1 = {
        "income": 60000,
        "loan_amount": 15000,
        "credit_history": 7,
        "explanation_methods": ["feature_importance"]
    }
    explanation_1 = call_explain_api(loan_data_1)
    print_explanation_summary(explanation_1)
    
    # Test Case 2: Basic loan with Shapley values
    print_header("Test Case 2: Basic Loan with Shapley Values")
    loan_data_2 = {
        "income": 60000,
        "loan_amount": 15000,
        "credit_history": 7,
        "explanation_methods": ["shapley_values"]
    }
    explanation_2 = call_explain_api(loan_data_2)
    print_explanation_summary(explanation_2)
    
    # Test Case 3: Edge case loan with counterfactual
    print_header("Test Case 3: Borderline Loan with Counterfactual")
    loan_data_3 = {
        "income": 50000,
        "loan_amount": 24000,  # Almost 50% of income
        "credit_history": 5,   # Mediocre credit
        "employment_years": 2,
        "debt_to_income_ratio": 0.35,
        "explanation_methods": ["counterfactual"]
    }
    explanation_3 = call_explain_api(loan_data_3)
    print_explanation_summary(explanation_3)
    
    # Test Case 4: Multiple explanation methods
    print_header("Test Case 4: Multiple Explanation Methods")
    loan_data_4 = {
        "income": 80000,
        "loan_amount": 30000,
        "credit_history": 8,
        "employment_years": 7,
        "debt_to_income_ratio": 0.2,
        "age": 42,
        "previous_defaults": 0,
        "dependents": 2,
        "explanation_methods": ["feature_importance", "shapley_values", "counterfactual"]
    }
    explanation_4 = call_explain_api(loan_data_4)
    print_explanation_summary(explanation_4)


if __name__ == "__main__":
    main()