def test_evaluation_metrics():
    """Test the evaluation metrics."""
    # Set device consistently
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    config = {
        'input_dim': 3,
        'hidden_dims': [32, 64, 128],
        'd_state': 16,
        'temporal_window': 4,
        'num_instances': 16
    }
    model = build_model(config).to(device)
    model.eval()  # Set to evaluation mode
    
    try:
        # Create dataloader
        transform = VideoSequenceAugmentation(
            img_size=(240, 320),
            normalize=True,
            train=False
        )
        
        # Test on multiple sequences
        sequences = ["breakdance", "camel", "car-roundabout"]
        all_predictions = []
        all_ground_truths = []
        sequence_names = []
        
        for sequence in sequences:
            try:
                dataloader = build_davis_dataloader(
                    root_path="/mnt/c/Datasets/DAVIS",  # Adjust to your path
                    split='val',
                    batch_size=1,
                    img_size=(240, 320),
                    sequence_length=4,
                    specific_sequence=sequence,
                    transform=transform
                )
                
                # Process one batch
                for batch in dataloader:
                    # Skip if no ground truth
                    if 'masks' not in batch:
                        continue
                    
                    # Move data to device
                    frames = batch['frames'].to(device)  # [B, T, C, H, W]
                    gt_masks = batch['masks'].to(device)  # [B, T, H, W]
                    
                    # Forward pass
                    with torch.no_grad():
                        outputs = model(frames)
                    
                    pred_masks = outputs['pred_masks']  # [B, T, N, H, W]
                    
                    # Save predictions and ground truth
                    all_predictions.append(pred_masks[0])  # Remove batch dimension
                    all_ground_truths.append(gt_masks[0])  # Remove batch dimension
                    sequence_names.append(sequence)
                    
                    # Only process one batch per sequence
                    break
            except Exception as e:
                print(f"Error processing sequence {sequence}: {str(e)}")
                continue
        
        # Skip evaluation if no sequences were processed
        if not all_predictions:
            print("No sequences processed, skipping evaluation")
            return False
        
        # Create evaluator
        evaluator = DAVISEvaluator()
        
        # Evaluate all sequences
        results = evaluator.evaluate(
            predictions=all_predictions,
            ground_truths=all_ground_truths,
            sequence_names=sequence_names
        )
        
        # Print results
        evaluator.print_results(results)
        
        print("Evaluation metrics test completed!")
        return True
        
    except Exception as e:
        print(f"Error during evaluation test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
