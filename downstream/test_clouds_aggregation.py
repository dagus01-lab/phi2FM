#!/usr/bin/env python3
"""
Test script to verify clouds label aggregation functionality.
"""

import numpy as np
import torch
import sys
sys.path.append('/home/gdaga/phi2FM/downstream')

from extract_samples_simple import aggregate_labels_for_task

def test_clouds_aggregation():
    """Test clouds task label aggregation."""
    print("Testing clouds label aggregation...")
    
    # Create test label with all 5 classes (0, 1, 2, 3, 4)
    h, w = 100, 100
    test_label = np.zeros((h, w), dtype=np.int64)
    
    # Fill different regions with different classes
    test_label[0:20, 0:20] = 0   # Class 0 (should become 0 - no clouds)
    test_label[0:20, 20:40] = 1  # Class 1 (should become 0 - no clouds)
    test_label[20:40, 0:20] = 2  # Class 2 (should become 1 - clouds)
    test_label[20:40, 20:40] = 3 # Class 3 (should become 1 - clouds)
    test_label[40:60, 0:20] = 4  # Class 4 (should become 1 - clouds)
    
    print(f"Original label classes: {np.unique(test_label)}")
    
    # Test aggregation
    aggregated = aggregate_labels_for_task(test_label, "clouds")
    
    # Convert to numpy for easier testing
    if isinstance(aggregated, torch.Tensor):
        aggregated_np = aggregated.cpu().numpy()
    else:
        aggregated_np = aggregated
    
    print(f"Aggregated label classes: {np.unique(aggregated_np)}")
    
    # Verify aggregation logic
    assert np.all(aggregated_np[0:20, 0:20] == 0), "Class 0 should become 0"
    assert np.all(aggregated_np[0:20, 20:40] == 0), "Class 1 should become 0"
    assert np.all(aggregated_np[20:40, 0:20] == 1), "Class 2 should become 1"
    assert np.all(aggregated_np[20:40, 20:40] == 1), "Class 3 should become 1"
    assert np.all(aggregated_np[40:60, 0:20] == 1), "Class 4 should become 1"
    
    # Test with torch tensor as well
    torch_label = torch.from_numpy(test_label)
    torch_aggregated = aggregate_labels_for_task(torch_label, "clouds")
    
    assert torch.equal(aggregated, torch_aggregated), "Torch and numpy results should be equal"
    
    print("✓ Clouds aggregation test passed!")
    print("  - Classes 0,1 → 0 (no clouds)")
    print("  - Classes 2,3,4 → 1 (clouds)")

def test_non_clouds_task():
    """Test that non-clouds tasks are not affected."""
    print("\nTesting non-clouds task (should not aggregate)...")
    
    test_label = np.array([[0, 1, 2], [3, 4, 0]], dtype=np.int64)
    original_copy = test_label.copy()
    
    # Test with a non-clouds task
    result = aggregate_labels_for_task(test_label, "fire")
    
    assert np.array_equal(result, original_copy), "Non-clouds tasks should not be modified"
    print("✓ Non-clouds task test passed!")

if __name__ == "__main__":
    test_clouds_aggregation()
    test_non_clouds_task()
    print("\n✓ All label aggregation tests passed!")