#!/usr/bin/env python3
"""
Unified Comparison Script for VINN and BehaviorRetrieval
Evaluates both methods on all target images and compares their retrieval behavior
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any

# Add parent directory to path to import from BehaviorRetrieval
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

################################################################################
#                    Action Vector Generation
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
#                    VINN Implementation
################################################################################

class VINNEvaluator:
    """VINN Evaluator - shows which samples are retrieved from database"""
    
    def __init__(self, model_dir='./vinn_target_models', device='cuda', k=16):
        self.device = device
        self.k = k
        self.model_dir = model_dir
        
        print("🔄 Loading VINN model...")
        
        # Load encoder
        self.encoder = resnet50(pretrained=False)
        self.encoder.fc = nn.Identity()
        
        encoder_path = f"{model_dir}/encoder.pth"
        if not os.path.exists(encoder_path):
            print(f"⚠️  VINN encoder not found at {encoder_path}")
            self.initialized = False
            return
            
        self.encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        self.encoder = self.encoder.to(device)
        self.encoder.eval()
        
        # Load database
        try:
            self.database_embeddings = np.load(f"{model_dir}/database_embeddings.npy")
            self.database_actions = np.load(f"{model_dir}/database_actions.npy")
        except:
            print("⚠️  VINN database not found")
            self.initialized = False
            return
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.initialized = True
        print(f"✅ VINN loaded: {len(self.database_embeddings)} demonstrations in database")
    
    def get_prediction_and_neighbors(self, query_image_path):
        """Get prediction and nearest neighbors info"""
        if not self.initialized:
            return None, None
        
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
            episode_id = (idx // 17) + 1
            step_id = (idx % 17) + 1
            
            neighbor_info.append({
                'rank': rank + 1,
                'database_index': idx,
                'distance': float(dist),
                'weight': float(weight),
                'episode': episode_id,
                'step': step_id,
                'action': self.database_actions[idx].tolist()
            })
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Update normalized weights
        for i, info in enumerate(neighbor_info):
            info['normalized_weight'] = float(weights[i])
        
        # Calculate weighted average action
        predicted_action = np.zeros_like(neighbor_actions[0])
        for weight, action in zip(weights, neighbor_actions):
            predicted_action += weight * action
        
        return predicted_action, neighbor_info

################################################################################
#                    BehaviorRetrieval Implementation
################################################################################

class BehaviorRetrievalEvaluator:
    """BehaviorRetrieval Evaluator - direct policy inference"""
    
    def __init__(self, model_dir='../BehaviorRetrieval/br_target_models', device='cuda'):
        self.device = device
        self.model_dir = model_dir
        
        print("🔄 Loading BehaviorRetrieval models...")
        
        # Load visual encoder
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 64)
        ).to(device)
        
        # Load policy network
        self.policy = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 7)
        ).to(device)
        
        # Load trained weights
        policy_path = f"{model_dir}/policy_target_training.pth"
        if not os.path.exists(policy_path):
            print(f"⚠️  BR policy not found at {policy_path}")
            self.initialized = False
            return
            
        self.policy.load_state_dict(torch.load(policy_path, map_location=device))
        
        # Load metadata
        self.metadata = {}
        metadata_path = f"{model_dir}/target_training_metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        
        self.visual_encoder.eval()
        self.policy.eval()
        self.initialized = True
        print("✅ BehaviorRetrieval models loaded")
    
    def predict_action(self, image_path):
        """Predict action using trained policy"""
        if not self.initialized:
            return None, None
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        
        # Resize to 84x84 as per BR paper
        image_resized = image.resize((84, 84), Image.LANCZOS)
        image_tensor = torch.FloatTensor(np.array(image_resized)).permute(2, 0, 1) / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Extract visual features
            visual_features = self.visual_encoder(image_tensor)
            
            # Predict action
            predicted_action = self.policy(visual_features).cpu().numpy()[0]
        
        return predicted_action, visual_features.cpu().numpy()[0]

################################################################################
#                    Comparison Functions
################################################################################

def compare_methods_on_sample(vinn_eval, br_eval, image_path, gt_action):
    """Compare both methods on a single sample"""
    
    result = {
        'image_path': str(image_path),
        'ground_truth': gt_action.tolist()
    }
    
    # VINN evaluation
    if vinn_eval.initialized:
        vinn_pred, vinn_neighbors = vinn_eval.get_prediction_and_neighbors(image_path)
        vinn_error = np.linalg.norm(vinn_pred - gt_action)
        
        result['vinn'] = {
            'prediction': vinn_pred.tolist(),
            'error': float(vinn_error),
            'top_neighbors': vinn_neighbors[:5] if vinn_neighbors else None  # Top 5 neighbors
        }
    else:
        result['vinn'] = {'error': 'Not initialized'}
    
    # BehaviorRetrieval evaluation
    if br_eval.initialized:
        br_pred, br_features = br_eval.predict_action(image_path)
        br_error = np.linalg.norm(br_pred - gt_action)
        
        result['behavior_retrieval'] = {
            'prediction': br_pred.tolist(),
            'error': float(br_error),
            'feature_norm': float(np.linalg.norm(br_features))
        }
    else:
        result['behavior_retrieval'] = {'error': 'Not initialized'}
    
    return result

def create_comparison_plots(results, output_dir):
    """Create visualization plots comparing both methods"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data for plotting
    vinn_errors = []
    br_errors = []
    episodes = []
    steps = []
    
    for r in results:
        if 'vinn' in r and isinstance(r['vinn']['error'], float):
            vinn_errors.append(r['vinn']['error'])
        else:
            vinn_errors.append(np.nan)
            
        if 'behavior_retrieval' in r and isinstance(r['behavior_retrieval']['error'], float):
            br_errors.append(r['behavior_retrieval']['error'])
        else:
            br_errors.append(np.nan)
            
        episodes.append(r['episode'])
        steps.append(r['step'])
    
    # Convert to numpy arrays
    vinn_errors = np.array(vinn_errors)
    br_errors = np.array(br_errors)
    
    # Filter out NaN values for statistics
    valid_vinn = vinn_errors[~np.isnan(vinn_errors)]
    valid_br = br_errors[~np.isnan(br_errors)]
    
    # 1. Error comparison scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(vinn_errors, br_errors, alpha=0.6)
    plt.plot([0, max(np.nanmax(vinn_errors), np.nanmax(br_errors))], 
             [0, max(np.nanmax(vinn_errors), np.nanmax(br_errors))], 
             'r--', alpha=0.5, label='Equal performance')
    plt.xlabel('VINN Error')
    plt.ylabel('BehaviorRetrieval Error')
    plt.title('Error Comparison: VINN vs BehaviorRetrieval')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'error_scatter.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Error distribution comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(valid_vinn, bins=30, alpha=0.5, label='VINN', density=True)
    axes[0].hist(valid_br, bins=30, alpha=0.5, label='BehaviorRetrieval', density=True)
    axes[0].set_xlabel('Error')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Error Distribution Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    data_to_plot = []
    labels = []
    if len(valid_vinn) > 0:
        data_to_plot.append(valid_vinn)
        labels.append('VINN')
    if len(valid_br) > 0:
        data_to_plot.append(valid_br)
        labels.append('BR')
    
    if data_to_plot:
        axes[1].boxplot(data_to_plot, labels=labels)
        axes[1].set_ylabel('Error')
        axes[1].set_title('Error Distribution Box Plot')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Per-episode performance
    episode_data = {}
    for r in results:
        ep = r['episode']
        if ep not in episode_data:
            episode_data[ep] = {'vinn': [], 'br': []}
        
        if 'vinn' in r and isinstance(r['vinn']['error'], float):
            episode_data[ep]['vinn'].append(r['vinn']['error'])
        if 'behavior_retrieval' in r and isinstance(r['behavior_retrieval']['error'], float):
            episode_data[ep]['br'].append(r['behavior_retrieval']['error'])
    
    episodes_sorted = sorted(episode_data.keys())
    vinn_means = [np.mean(episode_data[ep]['vinn']) if episode_data[ep]['vinn'] else np.nan 
                  for ep in episodes_sorted]
    br_means = [np.mean(episode_data[ep]['br']) if episode_data[ep]['br'] else np.nan 
                for ep in episodes_sorted]
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(episodes_sorted))
    width = 0.35
    
    plt.bar(x - width/2, vinn_means, width, label='VINN', alpha=0.8)
    plt.bar(x + width/2, br_means, width, label='BehaviorRetrieval', alpha=0.8)
    
    plt.xlabel('Episode')
    plt.ylabel('Average Error')
    plt.title('Per-Episode Performance Comparison')
    plt.xticks(x, episodes_sorted)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'per_episode_performance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualization plots saved to {output_dir}")

def analyze_retrieval_patterns(results, output_file):
    """Analyze retrieval patterns for VINN"""
    
    # Count neighbor retrieval frequencies
    neighbor_counts = {}
    top_neighbor_counts = {}
    
    for r in results:
        if 'vinn' in r and r['vinn'].get('top_neighbors'):
            for neighbor in r['vinn']['top_neighbors']:
                ep_step = f"Ep{neighbor['episode']}/Step{neighbor['step']:02d}"
                neighbor_counts[ep_step] = neighbor_counts.get(ep_step, 0) + 1
                
                # Count top-1 neighbors
                if neighbor['rank'] == 1:
                    top_neighbor_counts[ep_step] = top_neighbor_counts.get(ep_step, 0) + 1
    
    # Sort by frequency
    sorted_neighbors = sorted(neighbor_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_top_neighbors = sorted(top_neighbor_counts.items(), key=lambda x: x[1], reverse=True)
    
    analysis = {
        'total_retrievals': sum(neighbor_counts.values()),
        'unique_samples_retrieved': len(neighbor_counts),
        'most_frequently_retrieved': sorted_neighbors[:10],
        'most_frequently_top_neighbor': sorted_top_neighbors[:10]
    }
    
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"📊 Retrieval pattern analysis saved to {output_file}")

################################################################################
#                    Main Evaluation Function
################################################################################

def evaluate_all_samples(args):
    """Evaluate both methods on all target samples"""
    
    # Initialize evaluators
    vinn_eval = VINNEvaluator(
        model_dir=args.vinn_model_dir, 
        device=args.device, 
        k=args.k
    )
    
    br_eval = BehaviorRetrievalEvaluator(
        model_dir=args.br_model_dir, 
        device=args.device
    )
    
    all_results = []
    
    print("\n" + "="*80)
    print("🔍 UNIFIED COMPARISON: VINN vs BEHAVIOR RETRIEVAL")
    print("="*80)
    print(f"📁 Target directory: {args.target_dir}")
    print(f"🔍 VINN k-NN: {args.k}")
    print(f"💾 VINN models: {args.vinn_model_dir}")
    print(f"💾 BR models: {args.br_model_dir}")
    print("="*80)
    
    # Progress tracking
    total_episodes = 27
    total_steps = 17
    
    # Iterate through all episodes
    for episode_id in tqdm(range(1, 28), desc="Episodes", ncols=100):
        episode_path = Path(args.target_dir) / str(episode_id)
        if not episode_path.exists():
            continue
        
        for step in range(1, 18):
            # Try different image formats
            img_file = episode_path / f"{step:02d}.jpg"
            if not img_file.exists():
                img_file = episode_path / f"{step}.jpg"
            
            if not img_file.exists():
                continue
            
            # Get ground truth action
            gt_action_3d = get_action_vector(step, str(episode_id))
            gt_action_7d = np.array(gt_action_3d + [0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            
            # Compare both methods
            result = compare_methods_on_sample(vinn_eval, br_eval, str(img_file), gt_action_7d)
            result['episode'] = episode_id
            result['step'] = step
            
            all_results.append(result)
            
            # Print progress for verbose mode
            if args.verbose:
                vinn_err = result['vinn'].get('error', 'N/A')
                br_err = result['behavior_retrieval'].get('error', 'N/A')
                print(f"\nEp{episode_id}/Step{step:02d} - VINN: {vinn_err}, BR: {br_err}")
    
    # Calculate summary statistics
    vinn_errors = [r['vinn']['error'] for r in all_results 
                   if 'vinn' in r and isinstance(r['vinn']['error'], float)]
    br_errors = [r['behavior_retrieval']['error'] for r in all_results 
                 if 'behavior_retrieval' in r and isinstance(r['behavior_retrieval']['error'], float)]
    
    print("\n" + "="*80)
    print("📊 OVERALL COMPARISON RESULTS")
    print("="*80)
    print(f"Total samples evaluated: {len(all_results)}")
    
    if vinn_errors:
        print(f"\nVINN Performance:")
        print(f"  Mean error: {np.mean(vinn_errors):.6f}")
        print(f"  Std error: {np.std(vinn_errors):.6f}")
        print(f"  Min error: {np.min(vinn_errors):.6f}")
        print(f"  Max error: {np.max(vinn_errors):.6f}")
    
    if br_errors:
        print(f"\nBehaviorRetrieval Performance:")
        print(f"  Mean error: {np.mean(br_errors):.6f}")
        print(f"  Std error: {np.std(br_errors):.6f}")
        print(f"  Min error: {np.min(br_errors):.6f}")
        print(f"  Max error: {np.max(br_errors):.6f}")
    
    if vinn_errors and br_errors:
        # Compare performance
        vinn_better = sum(1 for v, b in zip(vinn_errors, br_errors) if v < b)
        br_better = sum(1 for v, b in zip(vinn_errors, br_errors) if b < v)
        ties = len(vinn_errors) - vinn_better - br_better
        
        print(f"\nHead-to-Head Comparison:")
        print(f"  VINN better: {vinn_better} ({vinn_better/len(vinn_errors)*100:.1f}%)")
        print(f"  BR better: {br_better} ({br_better/len(br_errors)*100:.1f}%)")
        print(f"  Ties: {ties} ({ties/len(vinn_errors)*100:.1f}%)")
    
    # Save detailed results
    output_file = args.output_file
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    # Create visualizations
    if args.visualize:
        create_comparison_plots(all_results, args.plot_dir)
    
    # Analyze retrieval patterns
    if vinn_eval.initialized:
        analyze_retrieval_patterns(all_results, 
                                 output_file.replace('.json', '_retrieval_analysis.json'))
    
    # Create summary report
    summary = {
        'total_samples': len(all_results),
        'vinn_performance': {
            'mean_error': float(np.mean(vinn_errors)) if vinn_errors else None,
            'std_error': float(np.std(vinn_errors)) if vinn_errors else None,
            'min_error': float(np.min(vinn_errors)) if vinn_errors else None,
            'max_error': float(np.max(vinn_errors)) if vinn_errors else None,
            'samples_evaluated': len(vinn_errors)
        },
        'br_performance': {
            'mean_error': float(np.mean(br_errors)) if br_errors else None,
            'std_error': float(np.std(br_errors)) if br_errors else None,
            'min_error': float(np.min(br_errors)) if br_errors else None,
            'max_error': float(np.max(br_errors)) if br_errors else None,
            'samples_evaluated': len(br_errors)
        }
    }
    
    if vinn_errors and br_errors:
        summary['comparison'] = {
            'vinn_better_count': vinn_better,
            'br_better_count': br_better,
            'tie_count': ties
        }
    
    summary_file = output_file.replace('.json', '_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"📊 Summary saved to: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description='Compare VINN and BehaviorRetrieval on all samples')
    
    # Paths
    parser.add_argument('--target_dir', type=str, 
                        default='/mnt/storage/owen/robot-dataset/rt-cache/raw/',
                        help='Path to target dataset')
    parser.add_argument('--vinn_model_dir', type=str, default='./vinn_target_models',
                        help='Path to VINN models')
    parser.add_argument('--br_model_dir', type=str, default='../BehaviorRetrieval/br_target_models',
                        help='Path to BehaviorRetrieval models')
    
    # Parameters
    parser.add_argument('--k', type=int, default=16,
                        help='Number of nearest neighbors for VINN')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed results')
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualization plots')
    parser.add_argument('--plot_dir', type=str, default='./comparison_plots',
                        help='Directory to save plots')
    parser.add_argument('--output_file', type=str, default='comparison_results.json',
                        help='Output file for results')
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.target_dir):
        print(f"❌ Target directory not found: {args.target_dir}")
        print("Using mock data path for testing...")
        args.target_dir = "./mock_target_data"
        
        # Create mock data for testing
        os.makedirs(args.target_dir, exist_ok=True)
        for ep in range(1, 4):  # Create 3 mock episodes
            ep_dir = os.path.join(args.target_dir, str(ep))
            os.makedirs(ep_dir, exist_ok=True)
            for step in range(1, 6):  # 5 steps each
                # Create a dummy image
                img = Image.new('RGB', (100, 100), color='red')
                img.save(os.path.join(ep_dir, f"{step:02d}.jpg"))
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, switching to CPU")
        args.device = 'cpu'
    
    # Run evaluation
    evaluate_all_samples(args)

if __name__ == '__main__':
    main()