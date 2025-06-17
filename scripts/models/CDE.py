import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from .NeuralCDE.metamodel import NeuralCDE
from .NeuralCDE.vector_fields import SingleHiddenLayer, FinalTanh
from .common import create_net, pearson_correlation, mask_from_lengths
from .AbstractContainer import AbstractContainer
import torch.autograd.functional as AF

class ModelContainer(AbstractContainer):
    def __init__(self, device):
        self.device = device
    
    def make_encoder(self, input_channels, hidden_channels, hidden_hidden_channels = 50, num_hidden_layers = 4):
        vector_field = FinalTanh(input_channels=input_channels, hidden_channels=hidden_channels,
                                                hidden_hidden_channels=hidden_hidden_channels,
                                                num_hidden_layers=num_hidden_layers)
        self.gen = NeuralCDE(func=vector_field, input_channels=input_channels, hidden_channels=hidden_channels, initial=True).to(self.device)
        return self.gen

    def make_decoder(self, latent_dim, output_channels, n_layers = 3, n_units = 100):
        self.pred = create_net(n_inputs = latent_dim, n_outputs = output_channels, n_layers = n_layers, n_units = n_units).to(self.device)
        return self.pred
    
    def loop(self, ob, dem, ac, scores, l, max_length, context_input, corr_coeff_param = 0.0, device = 'cuda', **kwargs):
        coefs = kwargs['coefs']
        idx = kwargs['idx']
        stabilization = kwargs['stabilization']
        lambda_reg = 0.01  # hyperparameter controlling strength of stiffness penalty
        
        targets = torch.cat((ob[:, 1:, :], torch.zeros(ob.shape[0], 1, ob.shape[-1]).to(device)), dim = 1)
        pred_mask = mask_from_lengths(l, max_length+1, device = device)[:, 1:]
        times = torch.arange(coefs[0].shape[1]+1, device=device).float()    
        
        coeffs_batch = [i[idx].to(device) for i in coefs]
        
        if stabilization == 'implicit_adams':                
            hidden = self.gen(times, 
                            coeffs_batch, 
                            final_index = -1, 
                            stream = True,
                            method='implicit_adams'
                            #   options={'step_size': 1.5} # Step size for implicit solver
                            )[:, :max_length, :]
        else:
            hidden = self.gen(times, 
                            coeffs_batch, 
                            final_index = -1, 
                            stream = True
                            )[:, :max_length, :]
            
        output = self.pred(hidden)
                
        total_loss = F.mse_loss(targets[pred_mask], output[pred_mask])
        mse_loss = total_loss.item()
        
        if corr_coeff_param > 0:
            corr_loss = pearson_correlation(hidden[pred_mask], scores[pred_mask], device=device).mean()
            total_loss += (-corr_coeff_param * corr_loss)
        
        # ---- Stiffness regularization ----
        if stabilization == 'l2_regularization':
            # Debug: Check hidden dimensions
            # print(f"\n[DEBUG] Starting stiffness regularization")
            # print(f"[DEBUG] Hidden tensor shape: {hidden.shape}")
            
            # Sample multiple time points
            time_samples = min(5, max_length)
            sample_indices = torch.linspace(0, max_length-1, time_samples, dtype=torch.long)
            h_samples = hidden[0, sample_indices].detach()
            # print(f"[DEBUG] Sampled {time_samples} time points, h_samples shape: {h_samples.shape}")
            
            total_stiffness = 0.0
            valid_samples = 0
            
            for i, h_sample in enumerate(h_samples):
                try:
                    h_sample = h_sample.clone().requires_grad_(True)
                    # print(f"\n[DEBUG] Processing sample {i+1}/{time_samples}")
                    # print(f"[DEBUG] h_sample shape: {h_sample.shape}")
                    
                    def f(h):
                        """Dimension-preserving vector field wrapper"""
                        # Add batch and time dimensions
                        h_reshaped = h.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_dim]
                        # print(f"[DEBUG] h_reshaped shape: {h_reshaped.shape}")
                        
                        # Compute vector field
                        out = self.gen.func(h_reshaped)
                        # print(f"[DEBUG] Raw vector field output shape: {out.shape}")
                        
                        # Handle different output cases
                        if out.dim() == 4:
                            # Case: [1, 1, hidden_dim, X]
                            # print("[DEBUG] Processing 4D output case")
                            if out.shape[2] == h.shape[0]:
                                # If middle dim matches hidden_dim, take diagonal
                                # print("[DEBUG] Using diagonal extraction")
                                out = torch.diagonal(out, dim1=2, dim2=3)  # [1, 1, min(hidden_dim,X)]
                                out = out[..., :h.shape[0]]  # Take first hidden_dim elements
                            else:
                                # Otherwise flatten
                                # print("[DEBUG] Flattening unusual 4D output")
                                out = out.mean(dim=2)  # [1, 1, X]
                        elif out.dim() == 3:
                            # Case: [1, 1, X]
                            # print("[DEBUG] Processing 3D output case")
                            pass
                        
                        # Final projection to match input dimension
                        if out.shape[-1] > h.shape[0]:
                            # print(f"[DEBUG] Truncating from {out.shape[-1]} to {h.shape[0]} channels")
                            out = out[..., :h.shape[0]]
                        elif out.shape[-1] < h.shape[0]:
                            # print(f"[DEBUG] Padding from {out.shape[-1]} to {h.shape[0]} channels")
                            out = F.pad(out, (0, h.shape[0] - out.shape[-1]))
                        
                        # print(f"[DEBUG] Final f(h) shape: {out.squeeze(0).squeeze(0).shape}")
                        return out.squeeze(0).squeeze(0)  # [hidden_dim]
                    
                    # Test function
                    test_out = f(h_sample)
                    assert test_out.shape == h_sample.shape
                        # f"Function output shape {test_out.shape} != input shape {h_sample.shape}"
                    
                    # Compute Jacobian using more stable approach
                    # print("[DEBUG] Computing Jacobian...")
                    with torch.enable_grad():
                        h_sample.requires_grad_(True)
                        f_h = f(h_sample)
                        
                        jacobian = []
                        for j in range(h_sample.size(0)):
                            # if j % 16 == 0:
                                # print(f"[DEBUG] Computing Jacobian row {j+1}/{h_sample.size(0)}")
                            grad_output = torch.zeros_like(f_h)
                            grad_output[j] = 1.0
                            grad_input = torch.autograd.grad(
                                outputs=f_h,
                                inputs=h_sample,
                                grad_outputs=grad_output,
                                retain_graph=True,
                                create_graph=True
                            )[0]
                            jacobian.append(grad_input)
                        
                        jacobian = torch.stack(jacobian, dim=0)
                        # print(f"[DEBUG] Jacobian shape: {jacobian.shape}")
                        
                        # Verify Jacobian is square
                        assert jacobian.shape[0] == jacobian.shape[1] == h_sample.shape[0], \
                            f"Jacobian shape {jacobian.shape} should be square ({h_sample.shape[0]}, {h_sample.shape[0]})"
                    
                    # Eigenvalue computation
                    # print("[DEBUG] Computing eigenvalues...")
                    jacobian = jacobian.detach().cpu()
                    try:
                        eigvals = torch.linalg.eigvals(jacobian)
                        real_eigvals = torch.real(eigvals)
                        
                        # Filter numerical issues
                        mask = ~torch.isnan(real_eigvals) & ~torch.isinf(real_eigvals)
                        if mask.sum() == 0:
                            # print("[WARNING] All eigenvalues were invalid!")
                            continue
                            
                        valid_eigvals = real_eigvals[mask]
                        stiffness = torch.max(torch.abs(valid_eigvals))
                        # print(f"[DEBUG] Max stiffness: {stiffness.item():.4f}")
                        
                        total_stiffness += stiffness
                        valid_samples += 1
                        
                    except RuntimeError as e:
                        print(f"[WARNING] Eigenvalue computation failed: {str(e)}")
                        continue
                        
                except Exception as e:
                    print(f"[ERROR] Processing sample {i} failed: {str(e)}")
                    continue
            
            if valid_samples > 0:
                avg_stiffness = total_stiffness / valid_samples
                # print(f"[DEBUG] Average stiffness: {avg_stiffness.item():.4f}")
                total_loss += lambda_reg * avg_stiffness
            else:
                print("[WARNING] No valid stiffness samples were computed!")
                
        return total_loss, mse_loss, hidden