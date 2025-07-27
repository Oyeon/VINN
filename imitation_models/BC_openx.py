import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
import wandb
import os

# Import the original BC models
from BC import TranslationModel, RotationModel, GripperModel

class BC:
    """BC class adapted for Open-X datasets while maintaining original logic"""
    
    def __init__(self, params):
        self.params = params
        self.params['representation'] = 0
        
        # Initialize device
        if self.params['gpu'] == 1:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        # Import necessary modules
        sys.path.append(params['root_dir'] + '../')
        from local_embedding_fix import LocalBYOLEmbeddingExtractor
        from train_BC_proper import OpenXDataset
        
        # Initialize encoder
        encoder = LocalBYOLEmbeddingExtractor(device=self.device)
        
        # Initialize wandb if needed
        if self.params['wandb'] == 1:
            if self.params['dataset'] == 'PushDataset':
                wandb.init(project = 'Push BC', entity="nyu_vinn")
                wandb.run.name = 'Push_BC_OpenX_' + str(self.params['pretrained'])
            elif self.params['dataset'] == 'StackDataset':
                wandb.init(project = 'Stack BC', entity="nyu_vinn")
                wandb.run.name = 'Stack_BC_OpenX_' + str(self.params['pretrained'])
            elif self.params['dataset'] == 'HandleData':
                wandb.init(project = 'Handle BC', entity="nyu_vinn")
                wandb.run.name = 'Handle_BC_OpenX_' + str(self.params['pretrained'])
        
        # Initialize loss tracking variables
        self.min_val_loss = float('inf')
        self.min_test_loss = float('inf')
        
        self.translation_loss_train = 0
        self.rotation_loss_train = 0
        self.gripper_loss_train = 0
        
        self.translation_loss_val = 0
        self.rotation_loss_val = 0
        self.gripper_loss_val = 0
        
        self.translation_loss_test = 0
        
        # Load datasets using Open-X
        print(f"Loading Open-X datasets for {self.params['dataset']} configuration...")
        
        # Training dataset (full)
        self.orig_img_data_train = OpenXDataset(self.params, encoder, partial=0.8)
        
        # Validation dataset (using remaining data)
        val_params = self.params.copy()
        self.img_data_val = OpenXDataset(val_params, encoder, partial=0.2)
        
        # For Push/Stack datasets, also create test set
        if self.params['dataset'] in ['PushDataset', 'StackDataset']:
            test_params = self.params.copy()
            self.img_data_test = OpenXDataset(test_params, encoder, partial=0.1)
            self.dataLoader_test = DataLoader(self.img_data_test, batch_size=self.params['batch_size'], 
                                            shuffle=True, pin_memory=True)
        
        # Initialize models with temporal dimension
        temporal_multiplier = self.params.get('t', 0) + 1
        
        if params['architecture'] == 'ResNet':
            input_dim = 2048 * temporal_multiplier
            self.translation_model = TranslationModel(input_dim).to(self.device)
            self.rotation_model = RotationModel(input_dim).to(self.device)
            self.gripper_model = GripperModel(input_dim).to(self.device)
        elif params['architecture'] == 'AlexNet':
            input_dim = 9216 * temporal_multiplier
            self.translation_model = TranslationModel(input_dim).to(self.device)
            self.rotation_model = RotationModel(input_dim).to(self.device)
            self.gripper_model = GripperModel(input_dim).to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            list(self.translation_model.parameters()) +
            list(self.rotation_model.parameters()) +
            list(self.gripper_model.parameters()), 
            lr=self.params['lr']
        )
        
        # Initialize data loaders
        self.dataLoader_val = DataLoader(self.img_data_val, batch_size=self.params['batch_size'], 
                                       shuffle=True, pin_memory=True)
        
        # Loss functions
        self.mseLoss = nn.MSELoss()
        self.ceLoss = nn.CrossEntropyLoss()
    
    def train(self):
        """Train the models (matching original logic)"""
        for epoch in tqdm(range(self.params['epochs'])):
            # Reset epoch losses
            if self.params['dataset'] in ['PushDataset', 'StackDataset']:
                self.translation_loss_train = 0
                self.translation_loss_val = 0
                self.translation_loss_test = 0
            else:  # HandleData
                self.translation_loss_train = 0
                self.rotation_loss_train = 0
                self.gripper_loss_train = 0
                
                self.translation_loss_val = 0
                self.rotation_loss_val = 0
                self.gripper_loss_val = 0
            
            # Training loop
            for i, data in enumerate(self.dataLoader_train, 0):
                self.optimizer.zero_grad()
                
                if self.params['dataset'] in ['PushDataset', 'StackDataset']:
                    representation, translation, path = data
                    
                    pred_translation = self.translation_model(representation.float().to(self.device))
                    loss = self.mseLoss(pred_translation, translation.float().to(self.device))
                    
                    self.translation_loss_train += loss.item() * representation.shape[0]
                
                else:  # HandleData
                    representation, translation, rotation, gripper, path = data
                    
                    pred_translation = self.translation_model(representation.float().to(self.device))
                    pred_rotation = self.rotation_model(representation.float().to(self.device))
                    pred_gripper = self.gripper_model(representation.float().to(self.device))
                    
                    translation_loss = self.mseLoss(pred_translation, translation.float().to(self.device))
                    rotation_loss = self.mseLoss(pred_rotation, rotation.float().to(self.device))
                    gripper_loss = self.ceLoss(pred_gripper, gripper.reshape(pred_gripper.shape[0],).to(self.device))
                    
                    self.translation_loss_train += translation_loss.item() * representation.shape[0]
                    self.rotation_loss_train += rotation_loss.item() * representation.shape[0]
                    self.gripper_loss_train += gripper_loss.item() * representation.shape[0]
                    
                    loss = translation_loss + rotation_loss + gripper_loss
                
                loss.backward()
                self.optimizer.step()
            
            # Normalize losses
            if self.params['dataset'] in ['PushDataset', 'StackDataset']:
                self.translation_loss_train /= len(self.img_data_train)
            else:
                self.translation_loss_train /= len(self.img_data_train)
                self.rotation_loss_train /= len(self.img_data_train)
                self.gripper_loss_train /= len(self.img_data_train)
            
            # Validation
            self.val()
            if self.params['dataset'] in ['PushDataset', 'StackDataset']:
                self.test()
            
            # Log to wandb
            if self.params['wandb'] == 1:
                self.wandb_publish()
            
            # Save checkpoints
            if epoch % 1000 == 0:
                self.save_model(epoch)
    
    def val(self):
        """Validation loop (matching original)"""
        for i, data in enumerate(self.dataLoader_val, 0):
            if self.params['dataset'] in ['PushDataset', 'StackDataset']:
                representation, translation, path = data
                
                pred_translation = self.translation_model(representation.float().to(self.device))
                loss = self.mseLoss(pred_translation, translation.float().to(self.device))
                
                self.translation_loss_val += loss.item() * representation.shape[0]
            
            else:  # HandleData
                representation, translation, rotation, gripper, path = data
                
                pred_translation = self.translation_model(representation.float().to(self.device))
                pred_rotation = self.rotation_model(representation.float().to(self.device))
                pred_gripper = self.gripper_model(representation.float().to(self.device))
                
                translation_loss = self.mseLoss(pred_translation, translation.float().to(self.device))
                rotation_loss = self.mseLoss(pred_rotation, rotation.float().to(self.device))
                gripper_loss = self.ceLoss(pred_gripper, gripper.reshape(pred_gripper.shape[0],).to(self.device))
                
                self.translation_loss_val += translation_loss.item() * representation.shape[0]
                self.rotation_loss_val += rotation_loss.item() * representation.shape[0]
                self.gripper_loss_val += gripper_loss.item() * representation.shape[0]
        
        # Normalize losses
        if self.params['dataset'] in ['PushDataset', 'StackDataset']:
            self.translation_loss_val /= len(self.img_data_val)
            self.min_val_loss = min(self.min_val_loss, self.translation_loss_val)
        else:
            self.translation_loss_val /= len(self.img_data_val)
            self.rotation_loss_val /= len(self.img_data_val)
            self.gripper_loss_val /= len(self.img_data_val)
            self.min_val_loss = min(self.min_val_loss, self.translation_loss_val)
    
    def test(self):
        """Test loop for Push/Stack datasets"""
        for i, data in enumerate(self.dataLoader_test, 0):
            representation, translation, path = data
            
            pred_translation = self.translation_model(representation.float().to(self.device))
            loss = self.mseLoss(pred_translation, translation.float().to(self.device))
            
            self.translation_loss_test += loss.item() * representation.shape[0]
        
        self.translation_loss_test /= len(self.img_data_test)
        self.min_test_loss = min(self.min_test_loss, self.translation_loss_test)
    
    def get_val_losses(self, fraction, times):
        """Get validation losses for fraction-based evaluation (matching original)"""
        losses = []
        
        for run in range(times):
            print(f"  Run {run+1}/{times} with fraction {fraction}")
            
            # Reset losses
            self.min_val_loss = float('inf')
            
            self.translation_loss_train = 0
            self.rotation_loss_train = 0
            self.gripper_loss_train = 0
            
            self.translation_loss_val = 0
            self.rotation_loss_val = 0
            self.gripper_loss_val = 0
            
            # Get subset of training data
            self.img_data_train = self.orig_img_data_train.get_subset(fraction)
            
            # Re-initialize models for this run
            temporal_multiplier = self.params.get('t', 0) + 1
            
            if self.params['architecture'] == 'ResNet':
                input_dim = 2048 * temporal_multiplier
                self.translation_model = TranslationModel(input_dim).to(self.device)
                self.rotation_model = RotationModel(input_dim).to(self.device)
                self.gripper_model = GripperModel(input_dim).to(self.device)
            elif self.params['architecture'] == 'AlexNet':
                input_dim = 9216 * temporal_multiplier
                self.translation_model = TranslationModel(input_dim).to(self.device)
                self.rotation_model = RotationModel(input_dim).to(self.device)
                self.gripper_model = GripperModel(input_dim).to(self.device)
            
            # Re-initialize optimizer
            self.optimizer = torch.optim.Adam(
                list(self.translation_model.parameters()) +
                list(self.rotation_model.parameters()) +
                list(self.gripper_model.parameters()), 
                lr=self.params['lr']
            )
            
            # Create data loader for subset
            self.dataLoader_train = DataLoader(self.img_data_train, batch_size=self.params['batch_size'], 
                                             shuffle=True, pin_memory=True)
            
            # Train on this subset
            self.train()
            
            print(f"    Translation val loss: {self.translation_loss_val:.4f}")
            losses.append(self.translation_loss_val)
        
        return losses
    
    def get_test_losses(self, fraction, times):
        """Get test losses for fraction-based evaluation (Push/Stack datasets)"""
        losses = []
        
        for run in range(times):
            print(f"  Run {run+1}/{times} with fraction {fraction}")
            
            # Reset losses
            self.min_val_loss = float('inf')
            self.min_test_loss = float('inf')
            
            self.translation_loss_train = 0
            self.translation_loss_val = 0
            self.translation_loss_test = 0
            
            # Get subset of training data
            self.img_data_train = self.orig_img_data_train.get_subset(fraction)
            
            # Re-initialize model for this run
            temporal_multiplier = self.params.get('t', 0) + 1
            
            if self.params['architecture'] == 'ResNet':
                input_dim = 2048 * temporal_multiplier
                self.translation_model = TranslationModel(input_dim).to(self.device)
            elif self.params['architecture'] == 'AlexNet':
                input_dim = 9216 * temporal_multiplier
                self.translation_model = TranslationModel(input_dim).to(self.device)
            
            # Re-initialize optimizer
            self.optimizer = torch.optim.Adam(self.translation_model.parameters(), lr=self.params['lr'])
            
            # Create data loader for subset
            self.dataLoader_train = DataLoader(self.img_data_train, batch_size=self.params['batch_size'], 
                                             shuffle=True, pin_memory=True)
            
            # Train on this subset
            self.train()
            
            print(f"    Test loss: {self.translation_loss_test:.4f}")
            losses.append(self.translation_loss_test)
        
        return losses
    
    def wandb_publish(self):
        """Publish metrics to wandb"""
        if self.params['dataset'] in ['PushDataset', 'StackDataset']:
            wandb.log({
                'train bc loss': self.translation_loss_train,
                'val bc loss': self.translation_loss_val,
                'test bc loss': self.translation_loss_test
            })
        else:  # HandleData
            wandb.log({
                'translation train': self.translation_loss_train,
                'rotation train': self.rotation_loss_train,
                'gripper train': self.gripper_loss_train,
                'translation val': self.translation_loss_val,
                'rotation val': self.rotation_loss_val,
                'gripper val': self.gripper_loss_val
            })
    
    def save_model(self, epoch):
        """Save model checkpoints"""
        os.makedirs(self.params['save_dir'], exist_ok=True)
        
        if self.params['dataset'] == 'PushDataset':
            torch.save({'model_state_dict': self.translation_model.state_dict()},
                      f"{self.params['save_dir']}/PushModel_translation_openx_{epoch}.pt")
        elif self.params['dataset'] == 'StackDataset':
            torch.save({'model_state_dict': self.translation_model.state_dict()},
                      f"{self.params['save_dir']}/StackModel_translation_openx_{epoch}.pt")
        else:  # HandleData
            torch.save({'model_state_dict': self.translation_model.state_dict()},
                      f"{self.params['save_dir']}/HandleModel_translation_openx_{epoch}.pt")
            torch.save({'model_state_dict': self.rotation_model.state_dict()},
                      f"{self.params['save_dir']}/HandleModel_rotation_openx_{epoch}.pt")
            torch.save({'model_state_dict': self.gripper_model.state_dict()},
                      f"{self.params['save_dir']}/HandleModel_gripper_openx_{epoch}.pt")