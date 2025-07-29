#!/usr/bin/env python3
"""
VINN Evaluation Script - Test on all target images
Shows which samples are retrieved for each query image
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
from pathlib import Path
import json
import argparse
from tqdm import tqdm

################################################################################
#                    Action Vector Generation (Your Original)
################################################################################

def get_action_vector(i: int, epi: str):
    """Your original Franka canned plan action vector generation"""
    def f(a,b,c,d,e): return \
        a if 1<=i<=5 else b if 6<=i<=8 else c if 9<=i<=12 else d if 13<=i<=17 else e
    _L1 = f([0, 0.035,0], [0,0,-0.055], [0,-0.02,0],  [0,0,-0.055], [0,0,0])
    _R1 = f([0,-0.035,0], [0,0,-0.055], [0, 0.02,0],  [0,0,-0.055], [0,0,0])
    _F1 = f([0.01,0,0],  [0,0,-0.055], [0,0.01,0],  [0,0,-0.055], [0,0,0])

    _L2 = f([0, 0.035,0], [0,0,-0.045], [-0.01, 0,0],  [0,0,-0.045], [0,0,0])
    _R2 = f([0,-0.035,0], [0,0,-0.045], [-0.01, 0,0],  [0,0,-0.045], [0,0,0])
    _F2 = f([0.02,0,0],  [0,0,-0.045], [-0.01, 0,0],  [0,0,-0.045], [0,0,0])
    
    _L3 = f([0, 0.035,0], [0,0,-0.055], [0, 0.01,0],  [0,0,-0.055], [0,0,0])
    _R3 = f([0,-0.035,0], [0,0,-0.055], [0, -0.01,0],  [0,0,-0.055], [0,0,0])
    _F3 = f([0.01,0,0],  [0,0,-0.055], [-0.01,0,0],  [0,0,-0.055], [0,0,0])

    families  = [[_L1,_L2,_L3], [_R1,_R2,_R3], [_F1,_F2,_F3]]

    try:
        eid = int(epi)
    except ValueError:
        return [0,0,0]
    if not 1<=eid<=28: return [0,0,0]
    fam  = (eid-1) % 3
    var  = ((eid-1)//3) % 3
    return families[fam][var]

################################################################################
#                    VINN Evaluator
################################################################################

class VINNEvaluator:
    """VINN Evaluator - shows which samples are retrieved from database"""
    
    def __init__(self, model_dir='./vinn_target_models', device='cuda', k=16):
        self.device = device
        self.k = k
        
        print("🔄 Loading VINN model...")
        
        # Load encoder
        self.encoder = resnet50(pretrained=False)
        self.encoder.fc = nn.Identity()
        
        encoder_path = f"{model_dir}/encoder.pth"
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Encoder not found at {encoder_path}. Please train VINN first.")
            
        self.encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        self.encoder = self.encoder.to(device)
        self.encoder.eval()
        
        # Load database
        self.database_embeddings = np.load(f"{model_dir}/database_embeddings.npy")
        self.database_actions = np.load(f"{model_dir}/database_actions.npy")
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✅ VINN loaded: {len(self.database_embeddings)} demonstrations in database")
        print(f"✅ Using k={k} nearest neighbors")
    
    def get_nearest_neighbors(self, query_image_path):
        """Get k nearest neighbors and their detailed info"""
        
        # Load and preprocess image
        query_image = Image.open(query_image_path).convert('RGB')
        query_tensor = self.transform(query_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            query_embedding = self.encoder(query_tensor).cpu().numpy()[0]
        
        # Compute distances to all demonstrations
        distances = []
        for i, demo_embedding in enumerate(self.database_embeddings):
            dist = np.linalg.norm(query_embedding - demo_embedding)
            distances.append((dist, i))
        
        # Sort by distance and get top k
        distances.sort(key=lambda x: x[0])
        top_k = distances[:self.k]
        
        # Process nearest neighbors
        weights = []
        neighbor_actions = []
        neighbor_info = []
        
        for rank, (dist, idx) in enumerate(top_k):
            # Calculate weight using Euclidean kernel
            weight = np.exp(-dist)
            weights.append(weight)
            neighbor_actions.append(self.database_actions[idx])
            
            # Determine which episode and step this neighbor is from
            # Assuming 17 steps per episode
            episode_id = (idx // 17) + 1
            step_id = (idx % 17) + 1
            
            neighbor_info.append({
                'rank': rank + 1,
                'database_index': idx,
                'distance': float(dist),
                'weight': float(weight),
                'episode': episode_id,
                'step': step_id,
                'action': self.database_actions[idx].tolist(),
                'image_path': f"Episode {episode_id}/Step {step_id:02d}"
            })
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Update normalized weights in neighbor info
        for i, info in enumerate(neighbor_info):
            info['normalized_weight'] = float(weights[i])
        
        # Calculate weighted average action
        predicted_action = np.zeros_like(neighbor_actions[0])
        for weight, action in zip(weights, neighbor_actions):
            predicted_action += weight * action
        
        return predicted_action, neighbor_info

def evaluate_all_vinn_samples(args):
    """Evaluate VINN on all target images"""
    
    # Initialize evaluator
    evaluator = VINNEvaluator(model_dir=args.model_dir, device=args.device, k=args.k)
    
    all_results = []
    
    print("\n" + "="*80)
    print("🔍 VINN EVALUATION ON ALL TARGET SAMPLES")
    print("="*80)
    print(f"📁 Target directory: {args.target_dir}")
    print(f"🔍 k-NN neighbors: {args.k}")
    print(f"💾 Model directory: {args.model_dir}")
    print("="*80)
    
    # Iterate through all episodes
    for episode_id in range(1, 28):  # Episodes 1-27
        episode_path = Path(args.target_dir) / str(episode_id)
        if not episode_path.exists():
            continue
        
        print(f"\n📂 Episode {episode_id}:")
        episode_results = []
        
        for step in range(1, 18):  # Steps 1-17
            # Try different image formats
            img_file = episode_path / f"{step:02d}.jpg"
            if not img_file.exists():
                img_file = episode_path / f"{step}.jpg"
            
            if not img_file.exists():
                continue
            
            # Get ground truth action
            gt_action_3d = get_action_vector(step, str(episode_id))
            gt_action_7d = np.array(gt_action_3d + [0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            
            # Get VINN prediction and neighbors
            predicted_action, neighbor_info = evaluator.get_nearest_neighbors(str(img_file))
            
            # Calculate error
            error = np.linalg.norm(predicted_action - gt_action_7d)
            
            # Store result
            result = {
                'episode': episode_id,
                'step': step,
                'image_path': str(img_file),
                'ground_truth_action': gt_action_7d.tolist(),
                'predicted_action': predicted_action.tolist(),
                'error': float(error),
                'neighbors': neighbor_info
            }
            
            episode_results.append(result)
            
            # Print summary
            if args.verbose:
                print(f"\n  Step {step:2d}:")
                print(f"    Ground truth: {gt_action_7d}")
                print(f"    Predicted:    {predicted_action}")
                print(f"    Error: {error:.6f}")
                print(f"    Top 3 neighbors:")
                for n in neighbor_info[:3]:
                    print(f"      - Episode {n['episode']}, Step {n['step']}: "
                          f"dist={n['distance']:.3f}, weight={n['normalized_weight']:.3f}")
            else:
                # Compact output
                top_neighbor = neighbor_info[0]
                print(f"  Step {step:2d}: error={error:.4f}, "
                      f"top neighbor: Ep{top_neighbor['episode']}/Step{top_neighbor['step']:02d} "
                      f"(dist={top_neighbor['distance']:.3f})")
        
        all_results.extend(episode_results)
        
        # Episode summary
        if episode_results:
            avg_error = np.mean([r['error'] for r in episode_results])
            print(f"  📊 Episode {episode_id} average error: {avg_error:.4f}")
    
    # Overall summary
    print("\n" + "="*80)
    print("📊 OVERALL VINN RESULTS")
    print("="*80)
    
    all_errors = [r['error'] for r in all_results]
    
    print(f"Total samples evaluated: {len(all_results)}")
    print(f"Mean error: {np.mean(all_errors):.6f}")
    print(f"Std error: {np.std(all_errors):.6f}")
    print(f"Min error: {np.min(all_errors):.6f}")
    print(f"Max error: {np.max(all_errors):.6f}")
    
    # Find best and worst predictions
    best = min(all_results, key=lambda x: x['error'])
    worst = max(all_results, key=lambda x: x['error'])
    
    print(f"\n🏆 Best prediction: Episode {best['episode']}, Step {best['step']} (error={best['error']:.6f})")
    print(f"❌ Worst prediction: Episode {worst['episode']}, Step {worst['step']} (error={worst['error']:.6f})")
    
    # Analyze neighbor patterns
    print("\n🔍 Neighbor Retrieval Analysis:")
    
    # Count how often each episode is retrieved as top neighbor
    top_neighbor_counts = {}
    for result in all_results:
        top_ep = result['neighbors'][0]['episode']
        top_neighbor_counts[top_ep] = top_neighbor_counts.get(top_ep, 0) + 1
    
    print("Most frequently retrieved episodes (as top neighbor):")
    for ep, count in sorted(top_neighbor_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  Episode {ep}: {count} times ({count/len(all_results)*100:.1f}%)")
    
    # Save detailed results
    output_file = args.output_file
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    # Create summary report
    summary = {
        'total_samples': len(all_results),
        'mean_error': float(np.mean(all_errors)),
        'std_error': float(np.std(all_errors)),
        'min_error': float(np.min(all_errors)),
        'max_error': float(np.max(all_errors)),
        'best_prediction': {
            'episode': best['episode'],
            'step': best['step'],
            'error': best['error']
        },
        'worst_prediction': {
            'episode': worst['episode'],
            'step': worst['step'],
            'error': worst['error']
        },
        'top_neighbor_frequency': top_neighbor_counts
    }
    
    summary_file = output_file.replace('.json', '_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📊 Summary saved to: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate VINN on all target samples')
    
    # Paths
    parser.add_argument('--target_dir', type=str, 
                        default='/mnt/storage/owen/robot-dataset/rt-cache/raw/',
                        help='Path to target dataset')
    parser.add_argument('--model_dir', type=str, default='./vinn_target_models',
                        help='Path to VINN models')
    
    # Parameters
    parser.add_argument('--k', type=int, default=16,
                        help='Number of nearest neighbors')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed results for each step')
    parser.add_argument('--output_file', type=str, default='vinn_evaluation_results.json',
                        help='Output file for detailed results')
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.target_dir):
        print(f"❌ Target directory not found: {args.target_dir}")
        return
    
    if not os.path.exists(args.model_dir):
        print(f"❌ Model directory not found: {args.model_dir}")
        print("Please train VINN first using: python vinn_target_training.py")
        return
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, switching to CPU")
        args.device = 'cpu'
    
    # Run evaluation
    evaluate_all_vinn_samples(args)

if __name__ == '__main__':
    main()