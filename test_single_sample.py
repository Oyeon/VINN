#!/usr/bin/env python3
"""
Test both VINN and BehaviorRetrieval on a single image sample
Shows which sample would be retrieved from VINN and the prediction from BehaviorRetrieval
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import json
import argparse

# Add parent directory to path
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
#                    Main Test Function
################################################################################

def test_single_sample(image_path, episode=None, step=None, k=16, device='cuda'):
    """Test both methods on a single sample"""
    
    print("="*80)
    print(f"🔍 TESTING SINGLE SAMPLE: {image_path}")
    print("="*80)
    
    # Initialize evaluators
    vinn_eval = VINNEvaluator(
        model_dir='./vinn_target_models', 
        device=device, 
        k=k
    )
    
    br_eval = BehaviorRetrievalEvaluator(
        model_dir='../BehaviorRetrieval/br_target_models', 
        device=device
    )
    
    # Get ground truth if episode and step are provided
    gt_action = None
    if episode is not None and step is not None:
        gt_action_3d = get_action_vector(step, str(episode))
        gt_action = np.array(gt_action_3d + [0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        print(f"📍 Ground truth action: {gt_action}")
    
    print("\n" + "-"*80)
    print("🔍 VINN RESULTS")
    print("-"*80)
    
    if vinn_eval.initialized:
        vinn_pred, vinn_neighbors = vinn_eval.get_prediction_and_neighbors(image_path)
        
        print(f"🎯 VINN prediction: {vinn_pred}")
        
        if gt_action is not None:
            vinn_error = np.linalg.norm(vinn_pred - gt_action)
            print(f"❌ VINN error: {vinn_error:.6f}")
        
        print(f"\n🏆 Top {min(10, len(vinn_neighbors))} retrieved neighbors:")
        for i, neighbor in enumerate(vinn_neighbors[:10]):
            print(f"  {i+1:2d}. Episode {neighbor['episode']:2d}, Step {neighbor['step']:2d} - "
                  f"dist: {neighbor['distance']:.4f}, weight: {neighbor['normalized_weight']:.4f}")
        
        # Show most frequently retrieved episodes
        episode_counts = {}
        for neighbor in vinn_neighbors:
            ep = neighbor['episode']
            episode_counts[ep] = episode_counts.get(ep, 0) + 1
        
        print(f"\n📊 Episode retrieval frequency (top {k}):")
        for ep, count in sorted(episode_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  Episode {ep}: {count} times ({count/k*100:.1f}%)")
    else:
        print("❌ VINN not initialized")
    
    print("\n" + "-"*80)
    print("🧠 BEHAVIOR RETRIEVAL RESULTS")
    print("-"*80)
    
    if br_eval.initialized:
        br_pred, br_features = br_eval.predict_action(image_path)
        
        print(f"🎯 BR prediction: {br_pred}")
        
        if gt_action is not None:
            br_error = np.linalg.norm(br_pred - gt_action)
            print(f"❌ BR error: {br_error:.6f}")
        
        print(f"🔢 Feature vector norm: {np.linalg.norm(br_features):.6f}")
    else:
        print("❌ BehaviorRetrieval not initialized")
    
    # Comparison
    if vinn_eval.initialized and br_eval.initialized and gt_action is not None:
        print("\n" + "-"*80)
        print("⚖️  COMPARISON")
        print("-"*80)
        
        vinn_error = np.linalg.norm(vinn_pred - gt_action)
        br_error = np.linalg.norm(br_pred - gt_action)
        
        if vinn_error < br_error:
            print(f"🏆 VINN performs better (error: {vinn_error:.6f} vs {br_error:.6f})")
        elif br_error < vinn_error:
            print(f"🏆 BehaviorRetrieval performs better (error: {br_error:.6f} vs {vinn_error:.6f})")
        else:
            print(f"🤝 Both methods perform equally (error: {vinn_error:.6f})")
        
        print(f"📏 Error difference: {abs(vinn_error - br_error):.6f}")

def main():
    parser = argparse.ArgumentParser(description='Test both methods on a single sample')
    
    parser.add_argument('image_path', type=str, help='Path to the query image')
    parser.add_argument('--episode', type=int, help='Episode number (for ground truth)')
    parser.add_argument('--step', type=int, help='Step number (for ground truth)')
    parser.add_argument('--k', type=int, default=16, help='Number of neighbors for VINN')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Validate image path
    if not os.path.exists(args.image_path):
        print(f"❌ Image not found: {args.image_path}")
        return
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, switching to CPU")
        args.device = 'cpu'
    
    # Run test
    test_single_sample(
        args.image_path, 
        args.episode, 
        args.step, 
        args.k, 
        args.device
    )

if __name__ == '__main__':
    main()