import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveLossFunctions:
    """Collection of loss functions suitable for sensitive regression values"""
    
    @staticmethod
    def huber_loss(predictions, targets, delta=1.0):
        """
        Huber Loss - Less sensitive to outliers than MSE, smoother than MAE
        Good for: Values with occasional outliers
        """
        return nn.HuberLoss(delta=delta)(predictions, targets)
    
    @staticmethod
    def smooth_l1_loss(predictions, targets, beta=1.0):
        """
        Smooth L1 Loss - Similar to Huber but with different parameterization
        Good for: Balancing between MSE and MAE behavior
        """
        return nn.SmoothL1Loss(beta=0.5)(predictions, targets)
    
    @staticmethod
    def log_cosh_loss(predictions, targets):
        """
        Log-Cosh Loss - Smooth approximation of MAE, less sensitive to outliers
        Good for: Sensitive values where you want smooth gradients
        """
        diff = predictions - targets
        return torch.mean(torch.log(torch.cosh(diff)))
    
    @staticmethod
    def quantile_loss(predictions, targets, quantile=0.5):
        """
        Quantile Loss - Asymmetric loss, good for imbalanced errors
        Good for: When over/under-prediction have different costs
        """
        errors = targets - predictions
        return torch.mean(torch.maximum(quantile * errors, (quantile - 1) * errors))
    
    @staticmethod
    def relative_loss(predictions, targets, epsilon=1e-8):
        """
        Relative Loss - Percentage-based, good for values of different scales
        Good for: Values spanning multiple orders of magnitude
        """
        return torch.mean(torch.abs((predictions - targets) / (torch.abs(targets) + epsilon)))
    
    @staticmethod
    def mape_loss(predictions, targets, epsilon=1e-8):
        """
        Mean Absolute Percentage Error Loss
        Good for: When relative errors matter more than absolute errors
        """
        return torch.mean(torch.abs((predictions - targets) / (torch.abs(targets) + epsilon))) * 100
    
    @staticmethod
    def focal_mse_loss(predictions, targets, alpha=2.0, gamma=1.0):
        """
        Focal MSE Loss - Focuses on hard examples
        Good for: When some samples are much harder to predict
        """
        mse = (predictions - targets) ** 2
        focal_weight = (mse + 1e-8) ** (gamma / 2)
        return torch.mean(alpha * focal_weight * mse)
    
    @staticmethod
    def adaptive_wing_loss(predictions, targets, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        """
        Adaptive Wing Loss - Good for continuous values with varying sensitivity
        Good for: High precision requirements with different error tolerances
        """
        diff = torch.abs(predictions - targets)
        A = omega * (1 / (1 + torch.pow(theta / epsilon, alpha - targets))) * (alpha - targets) * torch.pow(theta / epsilon, alpha - targets - 1) * (1 / epsilon)
        C = theta * A - omega * torch.log(1 + torch.pow(theta / epsilon, alpha - targets))
        
        loss = torch.where(
            diff < theta,
            omega * torch.log(1 + torch.pow(diff / epsilon, alpha - targets)),
            A * diff - C
        )
        return torch.mean(loss)
    
    @staticmethod
    def balanced_mse_mae_loss(predictions, targets, alpha=0.5):
        """
        Balanced MSE + MAE Loss
        Good for: Combining benefits of both MSE and MAE
        """
        mse = torch.mean((predictions - targets) ** 2)
        mae = torch.mean(torch.abs(predictions - targets))
        return alpha * mse + (1 - alpha) * mae
    
    @staticmethod
    def normalized_loss(predictions, targets, loss_fn='mse'):
        """
        Normalized Loss - Divides by target variance for scale-invariant loss
        Good for: Values with high variance or different scales
        """
        target_std = torch.std(targets) + 1e-8
        
        if loss_fn == 'mse':
            return torch.mean((predictions - targets) ** 2) / (target_std ** 2)
        elif loss_fn == 'mae':
            return torch.mean(torch.abs(predictions - targets)) / target_std
        else:
            raise ValueError("loss_fn must be 'mse' or 'mae'")


class DynamicLoss(nn.Module):
    """
    Dynamic Loss that adapts based on training progress
    Starts with one loss and gradually transitions to another
    """
    def __init__(self, loss_fn_1, loss_fn_2, transition_epochs=50):
        super().__init__()
        self.loss_fn_1 = loss_fn_1
        self.loss_fn_2 = loss_fn_2
        self.transition_epochs = transition_epochs
        self.current_epoch = 0
    
    def forward(self, predictions, targets):
        if self.current_epoch < self.transition_epochs:
            # Gradually transition from loss_fn_1 to loss_fn_2
            alpha = self.current_epoch / self.transition_epochs
            loss1 = self.loss_fn_1(predictions, targets)
            loss2 = self.loss_fn_2(predictions, targets)
            return (1 - alpha) * loss1 + alpha * loss2
        else:
            return self.loss_fn_2(predictions, targets)
    
    def step_epoch(self):
        self.current_epoch += 1