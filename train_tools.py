import os
import torch

def compute_gradient_stats_by_module(model):
    """
    Compute gradient statistics by module for visualization and debugging.
    
    Args:
        model: The model to analyze
        
    Returns:
        Dict containing gradient statistics for each module
    """
    # Handle DataParallel wrapper
    if hasattr(model, 'module'):
        model = model.module
    
    # Define module groups based on the model structure
    module_groups = {
        'embeddings': ['tri_token', 'reg_tokens'],
        'vn_encoding': ['vn_encoding_proj', 'vn_encoder_norm'],
        'transformer': ['transformer.layers'],
        'view_transformer_core': ['view_transformer.transformer'],
        'view_transformer_encoder': ['view_transformer.ray_map_patch_token', 'view_transformer.ray_map_encoder', 'view_transformer.ray_map_encoder_norm'],
        'view_transformer_output': ['view_transformer.out_dpt'],
        'rope_embeddings': ['rope_emb']
    }
    
    gradient_stats = {}
    
    for group_name, module_prefixes in module_groups.items():
        grad_tensors = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Check if this parameter belongs to the current module group
                for prefix in module_prefixes:
                    if name.startswith(prefix):
                        grad_tensors.append(param.grad.data.flatten())
                        break
        
        if grad_tensors:
            # Concatenate all gradient tensors for this module group
            all_grads = torch.cat(grad_tensors, dim=0)
            gradient_stats[group_name] = {
                'mean': all_grads.mean().item(),
                'max': all_grads.max().item(),
            }
    
    return gradient_stats


def save_checkpoint(model, optimizer, scheduler, epoch, loss, save_path):
    """Save model checkpoint in Hugging Face format"""
    os.makedirs(save_path, exist_ok=True)
    
    model_to_save = model.model.module if hasattr(model.model, 'module') else model.model
    model_to_save.save_pretrained(save_path)
    
    training_state = {
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }
    
    torch.save(training_state, os.path.join(save_path, 'training_state.pt'))
    
    print(f"Model saved to {save_path} in Hugging Face format")


def load_training_state(optimizer, scheduler, checkpoint_path):
    """Load model checkpoint from Hugging Face format"""

    training_state_path = os.path.join(checkpoint_path, 'training_state.pt')
    if os.path.exists(training_state_path):
        training_state = torch.load(training_state_path, map_location='cpu')
        
        optimizer.load_state_dict(training_state['optimizer_state_dict'])
        # scheduler.load_state_dict(training_state['scheduler_state_dict'])
        
        print(f"Resuming from epoch {training_state['epoch']} "
              f"with loss {training_state['loss']}")
        
        return (optimizer, scheduler, training_state['epoch'],
                training_state['loss'])
    else:
        print(f"No training state found at {training_state_path}")
        return optimizer, scheduler, 0, float('inf')


def load_epoch_from_training_state(checkpoint_path):
    """Load model checkpoint from Hugging Face format"""
    training_state_path = os.path.join(checkpoint_path, 'training_state.pt')
    if os.path.exists(training_state_path):
        training_state = torch.load(training_state_path, map_location='cpu')
        print(f"Resuming from epoch {training_state['epoch']} "
              f"with loss {training_state['loss']}")
        
        return training_state['epoch']
    else:
        return 0
    

def load_optimizer_state(optimizer, checkpoint_path, update_lr=1e-4):
    """Load optimizer state from checkpoint"""
    training_state_path = os.path.join(checkpoint_path, 'training_state.pt')
    if os.path.exists(training_state_path):
        training_state = torch.load(training_state_path, map_location='cpu')
        optimizer.load_state_dict(training_state['optimizer_state_dict'])

        # update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = update_lr
            param_group['initial_lr'] = update_lr
        
        print(f"Resuming from optimizer state")

        return optimizer
    else:
        print(f"No training state found at {training_state_path}")
        return optimizer
