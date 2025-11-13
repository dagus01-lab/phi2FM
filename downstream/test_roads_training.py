#!/usr/bin/env python3
"""
Test script to verify roads regression training can start without errors.
Tests dataloader, model initialization, and one forward/backward pass.
"""

import sys
import torch
from pathlib import Path

# Add parent directory to path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.load_data import load_data
from utils.training_utils import read_yaml
from models.model_Seco import seasonal_contrast

def test_roads_dataloader():
    """Test that dataloader can be created successfully for roads regression."""
    print("=" * 60)
    print("TEST 1: Loading roads regression dataloader")
    print("=" * 60)
    
    # Load configuration
    config_file = "args/finetune_FMs/roads/seco.yml"
    args = read_yaml(config_file)
    
    # Override n_shot to use smaller dataset for testing
    n_shot = 50  # Small number for quick testing
    
    # Set parameters
    batch_size = 4  # Small batch for testing
    dataset_folder = args.data_path_224_10m
    crop_images = False
    
    print(f"Dataset path: {dataset_folder}")
    print(f"n_shot: {n_shot}")
    print(f"output_channels: {args.output_channels}")
    
    try:
        # Load data
        weights, pos_weight, dl_train, dl_val, dl_test, dl_inference = load_data(
            dataset_folder,
            with_augmentations=False,
            num_workers=4,
            batch_size=batch_size,
            downstream_task='roads',
            model_name='seasonal_contrast',
            device='cpu',
            pad_bands=10,
            crop_images=crop_images,
            num_classes=args.output_channels,
            n=n_shot,
            weights_dir='test_roads'
        )
        
        print("✓ Dataloader created successfully!")
        print(f"  Train dataset size: {len(dl_train.dataset)}")
        print(f"  Val dataset size: {len(dl_val.dataset)}")
        print(f"  Test dataset size: {len(dl_test.dataset)}")
        
        # Test getting one batch
        batch = next(iter(dl_train))
        img = batch['img']
        label = batch['label']
        
        print(f"\n  Sample batch:")
        print(f"    Image shape: {img.shape}, dtype: {img.dtype}")
        print(f"    Label shape: {label.shape}, dtype: {label.dtype}")
        print(f"    Label range: [{label.min():.4f}, {label.max():.4f}]")
        
        # Verify labels are float (regression)
        assert label.dtype in [torch.float32, torch.float16], f"Expected float labels, got {label.dtype}"
        print("  ✓ Labels are floating point (regression task)")
        
        return True, dl_train, dl_val, dl_test, args
        
    except Exception as e:
        print(f"✗ Failed to create dataloader: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None, None, None


def test_model_initialization(args):
    """Test that model can be initialized with output_channels=1."""
    print("\n" + "=" * 60)
    print("TEST 2: Initializing model")
    print("=" * 60)
    
    try:
        device = torch.device('cpu')
        
        # Create model
        model = seasonal_contrast(
            input_channels=args.input_channels,
            output_channels=args.output_channels,
            num_layers=50,
            pretrained_path=args.pretrained_model_path,
            freeze_backbone=args.freeze_pretrained
        )
        
        model = model.to(device)
        
        print("✓ Model initialized successfully!")
        print(f"  Input channels: {args.input_channels}")
        print(f"  Output channels: {args.output_channels}")
        
        # Test forward pass with dummy input
        dummy_img = torch.randn(2, args.input_channels, 224, 224).to(device)
        with torch.no_grad():
            output = model(dummy_img)
            if hasattr(output, 'output'):
                output = output.output
        
        print(f"  Dummy forward pass:")
        print(f"    Input shape: {dummy_img.shape}")
        print(f"    Output shape: {output.shape}")
        
        # For regression with output_channels=1, output should be [B, 1, H, W]
        assert output.shape[1] == args.output_channels, f"Expected output_channels={args.output_channels}, got {output.shape[1]}"
        print("  ✓ Forward pass successful")
        
        return True, model
        
    except Exception as e:
        print(f"✗ Failed to initialize model: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_training_step(model, dl_train):
    """Test one training step (forward + backward pass)."""
    print("\n" + "=" * 60)
    print("TEST 3: Running one training step")
    print("=" * 60)
    
    try:
        device = torch.device('cpu')
        model = model.to(device)
        model.train()
        
        # Setup optimizer and loss
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
        criterion = torch.nn.MSELoss()
        
        # Get one batch
        batch = next(iter(dl_train))
        images = batch['img'].to(device)
        labels = batch['label'].to(device)
        
        print(f"  Batch shapes:")
        print(f"    Images: {images.shape}")
        print(f"    Labels: {labels.shape}")
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        if hasattr(outputs, 'output'):
            outputs = outputs.output
        
        print(f"    Outputs: {outputs.shape}")
        
        # Verify shapes match
        assert outputs.shape == labels.shape, f"Shape mismatch: outputs {outputs.shape} vs labels {labels.shape}"
        
        # Compute loss
        loss = criterion(outputs, labels)
        print(f"\n  Loss: {loss.item():.6f}")
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        print("  ✓ Training step completed successfully!")
        print("  ✓ Forward pass works")
        print("  ✓ Loss computation works")
        print("  ✓ Backward pass works")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed training step: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("ROADS REGRESSION TRAINING VERIFICATION")
    print("=" * 60 + "\n")
    
    # Test 1: Dataloader
    success1, dl_train, dl_val, dl_test, args = test_roads_dataloader()
    if not success1:
        print("\n❌ FAILED: Dataloader test failed")
        return False
    
    # Test 2: Model initialization
    success2, model = test_model_initialization(args)
    if not success2:
        print("\n❌ FAILED: Model initialization failed")
        return False
    
    # Test 3: Training step
    success3 = test_training_step(model, dl_train)
    if not success3:
        print("\n❌ FAILED: Training step failed")
        return False
    
    # All tests passed
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nThe roads regression training configuration is ready to use.")
    print("You can now run the full training with:")
    print("  python training_script.py --config args/finetune_FMs/roads/seco.yml")
    print("=" * 60 + "\n")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
