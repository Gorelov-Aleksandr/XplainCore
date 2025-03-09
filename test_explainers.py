#!/usr/bin/env python3
"""
Test script for the XAI service with various explanation methods.
This script demonstrates how to call the API with different loan scenarios
and explanation methods.
"""
import json
import requests
import time
from pprint import pprint

# Configuration
BASE_URL = "http://localhost:5000"
API_KEY = "XAI-dev-key-2023"

# Default headers
headers = {
    "accept": "application/json",
    "Content-Type": "application/json",
    "X-API-Key": API_KEY
}

def call_explain_api(loan_data, methods=None):
    """Call the explain API with the given loan data and methods."""
    url = f"{BASE_URL}/explain"
    
    # Add explanation methods if provided
    if methods:
        loan_data["explanation_methods"] = methods
    
    # Make the API call
    response = requests.post(url, headers=headers, json=loan_data)
    
    # Process the response
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)

def print_explanation_summary(explanation):
    """Print a summary of the explanation."""
    if not explanation:
        print("No explanation available")
        return
    
    request_id = explanation.get("request_id", "Unknown")
    prediction = explanation.get("prediction", {})
    confidence = explanation.get("confidence", {})
    explanations = explanation.get("explanations", [])
    computation_time = explanation.get("computation_time", {})
    
    print(f"Request ID: {request_id}")
    print(f"Prediction: {prediction.get('decision', 'Unknown')} (Score: {prediction.get('score', 'N/A')})")
    print(f"Confidence: {confidence.get('overall_score', 'Unknown')}")
    print(f"Computation Time: {computation_time.get('total', 0):.4f} seconds")
    print(f"Number of Explanations: {len(explanations)}")
    
    # Print details for each explanation method
    for i, exp in enumerate(explanations):
        method = exp.get("method", "Unknown")
        print(f"\n  Explanation {i+1}: {method}")
        
        # Print feature importance if available
        if exp.get("feature_importance"):
            print("    Feature Importance:")
            for feature, importance in exp["feature_importance"].items():
                print(f"      {feature}: {importance:.4f}")
        
        # Print decision rules if available
        if exp.get("decision_rules"):
            print("    Decision Rules:")
            for rule in exp["decision_rules"]:
                print(f"      - {rule}")
        
        # Print counterfactuals if available
        if exp.get("counterfactuals"):
            print("    Counterfactuals:")
            for j, cf in enumerate(exp["counterfactuals"]):
                print(f"      Counterfactual {j+1}: {cf.get('explanation', 'No explanation')}")
        
        # Print feature interactions if available
        if exp.get("feature_interactions"):
            print("    Feature Interactions: Available")
        
        # Print visualizations if available
        visualizations = exp.get("visualizations", [])
        print(f"    Visualizations: {len(visualizations)} available")

def main():
    """Main function to test the XAI service."""
    print_header("XAI Service Testing")
    
    # Test Case 1: Approved loan with feature importance
    print_header("Test Case 1: Approved Loan (Feature Importance)")
    loan_data_1 = {
        "income": 80000,
        "loan_amount": 20000,
        "credit_history": 8,
        "employment_years": 5,
        "debt_to_income_ratio": 0.2
    }
    result_1 = call_explain_api(loan_data_1, ["feature_importance"])
    print_explanation_summary(result_1)
    
    # Test Case 2: Denied loan with Shapley values
    print_header("Test Case 2: Denied Loan (Shapley Values)")
    loan_data_2 = {
        "income": 50000,
        "loan_amount": 20000,
        "credit_history": 4,
        "employment_years": 2,
        "debt_to_income_ratio": 0.4
    }
    result_2 = call_explain_api(loan_data_2, ["shapley_values"])
    print_explanation_summary(result_2)
    
    # Test Case 3: Borderline loan with counterfactual explanation
    print_header("Test Case 3: Borderline Loan (Counterfactual)")
    loan_data_3 = {
        "income": 65000,
        "loan_amount": 25000,
        "credit_history": 6,
        "employment_years": 3,
        "debt_to_income_ratio": 0.35
    }
    result_3 = call_explain_api(loan_data_3, ["counterfactual"])
    print_explanation_summary(result_3)
    
    # Test Case 4: Multi-method explanation
    print_header("Test Case 4: All Explanation Methods")
    loan_data_4 = {
        "income": 70000,
        "loan_amount": 22000,
        "credit_history": 7,
        "employment_years": 4,
        "debt_to_income_ratio": 0.25
    }
    result_4 = call_explain_api(loan_data_4, ["feature_importance", "shapley_values", "counterfactual"])
    print_explanation_summary(result_4)
    
    # Print the full result for the multi-method explanation if needed
    # print_header("Full Result for Test Case 4")
    # pprint(result_4)

if __name__ == "__main__":
    main()