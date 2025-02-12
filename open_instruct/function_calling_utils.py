from typing import List, Dict, Any


def is_function_calls_equivalent(
    actual_calls: List[Dict[str, Any]],
    expected_calls: List[Dict[str, Any]]
) -> bool:
    """
    Check if two lists of function calls are equivalent.
    
    Args:
        actual_calls: List of actual function calls
        expected_calls: List of expected function calls
    
    Returns:
        bool: True if calls are equivalent
    """
    if len(actual_calls) != len(expected_calls):
        return False
        
    for actual, expected in zip(actual_calls, expected_calls):
        # Check basic structure
        if not isinstance(actual, dict) or not isinstance(expected, dict):
            return False
            
        # Check required fields
        if "name" not in actual or "arguments" not in actual:
            return False
            
        # Compare function names
        if actual["name"] != expected["name"]:
            return False
            
        # Compare arguments
        actual_args = actual["arguments"]
        expected_args = expected["arguments"]
        
        if not isinstance(actual_args, dict) or not isinstance(expected_args, dict):
            return False
            
        # Check if all expected arguments are present with correct values
        for key, value in expected_args.items():
            if key not in actual_args or actual_args[key] != value:
                return False
                
    return True