import numpy as np
import torch
from time import time
import logging
from tqdm import tqdm as log_progress

from .PINN_utils import PINN_utils

class PINN(PINN_utils):
    
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)       
    
    def move_batch_to_device(self, X_batch):
        """Move batch data to the appropriate device (GPU/CPU)"""
        X_batch_device = {}
        for key, value in X_batch.items():
            if isinstance(value, tuple):
                # Handle nested tuples
                X_batch_device[key] = tuple(
                    tuple(v.to(self.device) if torch.is_tensor(v) else v for v in item) if isinstance(item, tuple)
                    else item.to(self.device) if torch.is_tensor(item) else item
                    for item in value
                )
            elif torch.is_tensor(value):
                X_batch_device[key] = value.to(self.device)
            else:
                X_batch_device[key] = value
        return X_batch_device
    
    def get_loss(self, X_batch, model, w, validation=False):
        # Move batch to device
        X_batch = self.move_batch_to_device(X_batch)
        loss = 0.0
        L = self.PDE.get_loss(X_batch, model, validation=validation)
        for t in self.mesh.domain_mesh_names:
            loss += w[t]*L[t]
        return loss,L

    def get_grad_loss(self,X_batch, model, w):
        loss,L = self.get_loss(X_batch, model, w)
        loss.backward()
        g = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in model.parameters()]
        return loss, L, g
    

    def train_sgd(self, X_d, X_v):

        def train_step(X_batch, ws):
            self.optimizer.zero_grad()
            loss, L_loss, grad_theta = self.get_grad_loss(X_batch, self.model, ws)
            self.optimizer.step()
            L = [loss,L_loss]
            return L
        
        def calculate_validation_loss(X_v):
            # Note: We need gradients for residual computation, so don't use torch.no_grad()
            # The validation flag prevents parameter updates
            loss,L_loss = self.get_loss(X_v,self.model,self.w, validation=True)
            L = [loss,L_loss]
            return L

        for i in range(self.N_iters-self.iter):

            if self.sample_method == 'random_sample':
                X_d = self.get_batches(self.sample_method)

            L = train_step(X_d, ws=self.w) 
            L_v = calculate_validation_loss(X_v)
            self.complete_callback(L,L_v)


    def train_newton(self, X_batch, X_batch_val):

        def train_step(X_batch, X_batch_val):
            # Create LBFGS optimizer using YAML configuration
            optimizer_lbfgs = torch.optim.LBFGS(
                self.model.parameters(),
                max_iter=self.optimizer_2_opts['maxiter'],
                max_eval=self.optimizer_2_opts.get('maxfun', None),
                tolerance_grad=self.optimizer_2_opts.get('gtol', 1e-5),
                tolerance_change=self.optimizer_2_opts['ftol'],
                history_size=self.optimizer_2_opts['maxcor'],
                line_search_fn='strong_wolfe'
            )
            
            def closure():
                """Closure function for LBFGS - computes loss and gradients"""
                optimizer_lbfgs.zero_grad()
                loss, L_loss = self.get_loss(X_batch, self.model, self.w)
                loss.backward()
                return loss
            
            # Run LBFGS optimization step (calls closure multiple times internally)
            optimizer_lbfgs.step(closure)
            
            # Callback once after the full LBFGS step completes
            loss, L_loss = self.get_loss(X_batch, self.model, self.w)
            L_v = self.get_loss(X_batch_val, self.model, self.w, validation=True)
            L = [loss, L_loss]
            self.complete_callback(L, L_v)

        for i in range(self.N_steps_2):
            if self.sample_method == 'random_sample':
                X_batch = self.get_batches(self.sample_method)
            
            train_step(X_batch, X_batch_val)
            
            
    def main_loop(self, N=1000, N2=0):
        
        self.N_iters = N
        self.N_steps_2 = N2

        if self.use_optimizer_2:
            # Each LBFGS step produces one callback (not maxiter callbacks)
            self.N_iters_2 = self.N_steps_2 

        N_total = self.N_iters + self.N_iters_2
        self.pbar = log_progress(range(N_total))
        self.pbar.update(self.iter)
        self.pbar.refresh()

        if self.starting_point == 'new':
            self.create_losses_arrays(N_total)
        if N_total > len(self.losses['TL']):
            self.extend_losses_arrays(N_total)
            
        self.optimizer = self.create_optimizer(self.starting_point)
        
        X_v = self.get_batches('full_batch', validation=True)
        X_d = self.get_batches(self.sample_method)

        self.initialize_indicators()

        self.train_sgd(X_d, X_v)

        if self.use_optimizer_2:
            self.train_newton(X_d,X_v)


    def complete_callback(self,L,L_v):

        self.iter+=1
        self.checkers_iterations()
        self.calculate_Indicators(self.calc_Indicator_now)
        self.callback(L,L_v)
        self.check_adapt_new_weights(self.adapt_w_now)

        self.pbar.update()
        if self.iter % 2 == 0:
            opt_name = self.optimizer_name if self.iter<=self.N_iters else self.optimizer_2_name
            self.pbar.set_description("{} loop, G_solv: {:6.3}, Loss: {:6.4e}".format(opt_name, self.current_G_solv, self.current_loss))  


    def check_adapt_new_weights(self,adapt_now):
        
        if adapt_now:
            X_d = self.get_batches(self.sample_method)
            self.modify_weights_by(self.model,X_d) 
            
    def modify_weights_by(self,model,X_domain):
        
        L = dict()
        if self.adapt_w_method == 'gradients':
            _,L_loss = self.get_loss(X_domain, model, self.w)

            for t in self.mesh.domain_mesh_names:
                loss = L_loss[t]
                model.zero_grad()
                loss.backward(retain_graph=True)
                grads = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in model.parameters()]
                gradient_norm = torch.sqrt(sum([torch.sum(g**2) for g in grads]))
                L[t] = gradient_norm.item()

        elif self.adapt_w_method == 'values':
            with torch.no_grad():
                _,L_temp = self.get_loss(X_domain, model, self.w)
            L = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in L_temp.items()}

        eps = 1e-9
        loss_wo_w = sum(L.values())
        for t in self.mesh.domain_mesh_names:
            w = float(loss_wo_w/(L[t]+eps))
            self.w[t] = self.alpha_w*self.w[t] + (1-self.alpha_w)*w  

    def calculate_Indicators(self,calc_now):
        if calc_now:
            if self.Indicators['G_solv']:
                with torch.no_grad():
                    G_solv_tensor = self.PDE.get_solvation_energy(self.model)
                    self.current_G_solv = G_solv_tensor.cpu().numpy() if isinstance(G_solv_tensor, torch.Tensor) else G_solv_tensor
                self.G_solv_hist[str(self.iter)] = self.current_G_solv   

            if self.Indicators['L2_error_phi']:
                with torch.no_grad():
                    phi_pinn = self.PDE.get_phi_interface_verts(self.model,value='react')[0]
                    phi_pinn_np = phi_pinn.cpu().numpy() if isinstance(phi_pinn, torch.Tensor) else phi_pinn
                phi_dif = (phi_pinn_np.reshape(-1,1) - self.phi_known_L2.reshape(-1,1))
                error = np.sqrt(np.sum(phi_dif**2)/np.sum(self.phi_known_L2.reshape(-1,1)**2))

                self.current_L2_error = error
                self.L2_error_hist[str(self.iter)] = self.current_L2_error 


    def solve(self,N=1000, N2=0, save_model=0, Indicators_iter=100, Indicators=dict(G_solv=True)):

        self.save_model_iter = save_model if save_model != 0 else N
        self.Indicators_iter = Indicators_iter

        self.Indicators = {'G_solv': True, 'L2_error_phi': False}
        for key in self.Indicators:
            if key in Indicators:
                self.Indicators[key] = Indicators[key]

        t0 = time()
        
        self.main_loop(N,N2)

        import os
        dir_save = os.path.join(self.results_path,'iterations',f'iter_{self.iter}')
        self.save_model(dir_save)

        logger = logging.getLogger(__name__)
        logger.info(f' Iterations: {self.iter}')
        logger.info(" Loss: {:6.4e}".format(self.losses['TL'][self.iter-1]))
        print('\nComputation time: {} minutes'.format(int((time()-t0)/60)))
        logger.info('Computation time: {} minutes'.format(int((time()-t0)/60)))
