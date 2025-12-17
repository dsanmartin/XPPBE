import numpy as np
import torch
from scipy import special as sp
from functools import wraps

class Solution_utils(): 

    qe = 1.60217663e-19
    eps0 = 8.8541878128e-12     
    kb = 1.380649e-23              
    Na = 6.02214076e23
    ang_to_m = 1e-10
    cal2j = 4.184

    pi = np.pi

    apbs_created = False
    pbj_created = False
    pbj_mesh_density = 5
    pbj_mesh_generator = 'msms'

    def phi_known(self,function,field,X,flag,*args,**kwargs):
        if function == 'Spherical_Harmonics':
            phi_values = self.Spherical_Harmonics(X, flag,*args,**kwargs)
            if flag=='solvent':
                phi_values -= self.G(X)[:,0]
        elif function == 'G_Yukawa':
            phi_values = self.G_Yukawa(X)[:,0] - self.G(X)[:,0]
        elif function == 'analytic_Born_Ion':
            # Convert to numpy if tensor
            X_np = X.detach().cpu().numpy() if torch.is_tensor(X) else X
            phi_values = self.analytic_Born_Ion(np.linalg.norm(X_np, axis=1) ,*args,**kwargs)
        elif function == 'PBJ':
            phi_values = self.pbj_solution(X, flag)
            if flag=='solvent':
                phi_values -= self.G(X)[:,0]
        elif function == 'APBS':
            phi_values = self.apbs_solution(X)
        
        if field == 'react':
            return np.array(phi_values)
        elif field == 'phi':
            return np.array(phi_values + self.G(X)[:,0])


    def adim_function(function):
        @wraps(function)
        def wrapper(self,*kargs,**kwargs):
            ans = function(self,*kargs,**kwargs)
            beta_inv = self.beta**-1
            # Handle tensor/numpy compatibility
            if torch.is_tensor(ans):
                return ans * beta_inv
            else:
                # Convert beta to numpy scalar for numpy operations
                beta_inv_np = beta_inv.detach().cpu().numpy() if torch.is_tensor(beta_inv) else beta_inv
                return ans * beta_inv_np
        return wrapper

    @adim_function
    def G(self,X):
        # Convert to tensor if numpy
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        x_qs = torch.from_numpy(self.x_qs).float() if isinstance(self.x_qs, np.ndarray) else self.x_qs
        qs = torch.from_numpy(self.qs).float() if isinstance(self.qs, np.ndarray) else self.qs
        
        # Move to same device as X
        x_qs = x_qs.to(X.device)
        qs = qs.to(X.device)
        
        r_vec_expanded = X.unsqueeze(1)
        x_qs_expanded = x_qs.unsqueeze(0)
        r_diff = r_vec_expanded - x_qs_expanded
        r = torch.sqrt(torch.sum(r_diff**2, dim=2))
        q_over_r = qs / r
        total_sum = torch.sum(q_over_r, dim=1)
        result = (1 / (self.epsilon_1 * 4 * self.pi)) * total_sum
        result = result.unsqueeze(1)
        return result
    
    @adim_function
    def dG_n(self,X,n):
        # Convert to tensor if numpy
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        if isinstance(n, np.ndarray):
            n = torch.from_numpy(n).float()
        x_qs = torch.from_numpy(self.x_qs).float() if isinstance(self.x_qs, np.ndarray) else self.x_qs
        qs = torch.from_numpy(self.qs).float() if isinstance(self.qs, np.ndarray) else self.qs
        
        # Move to same device as X
        x_qs = x_qs.to(X.device)
        qs = qs.to(X.device)
        n = n.to(X.device)
        
        r_vec_expanded = X.unsqueeze(1)
        x_qs_expanded = x_qs.unsqueeze(0)
        r_diff = r_vec_expanded - x_qs_expanded
        r = torch.sqrt(torch.sum(r_diff**2, dim=2))
        dg_dr = qs / (r**3) * (-1 / (self.epsilon_1 * 4 * self.pi)) * (1/2)
        dx = dg_dr * 2*r_diff[:, :, 0]
        dy = dg_dr * 2*r_diff[:, :, 1]
        dz = dg_dr * 2*r_diff[:, :, 2]
        dx_sum = torch.sum(dx, dim=1)
        dy_sum = torch.sum(dy, dim=1)
        dz_sum = torch.sum(dz, dim=1)
        dg_dn = n[:, 0] * dx_sum + n[:, 1] * dy_sum + n[:, 2] * dz_sum
        return dg_dn.reshape(-1, 1) 
    

    def source(self,X):
        x_qs = self.x_qs.to(X.device) if torch.is_tensor(self.x_qs) else torch.from_numpy(self.x_qs).float().to(X.device)
        qs = self.qs.to(X.device) if torch.is_tensor(self.qs) else torch.from_numpy(self.qs).float().to(X.device)
        
        r_vec_expanded = X.unsqueeze(1)
        x_qs_expanded = x_qs.unsqueeze(0)
        r_diff = r_vec_expanded - x_qs_expanded
        r2 = torch.sum(r_diff**2, dim=2)
        delta = torch.exp((-1/(2*self.sigma**2))*r2)
        q_times_delta = qs*delta
        total_sum = torch.sum(q_times_delta, dim=1)
        normalizer = (1/((2*self.pi)**(3.0/2)*self.sigma**3))
        result = (-1/self.epsilon_1)*normalizer * total_sum
        result = result.unsqueeze(1)
        return result
        
    @adim_function
    def G_Yukawa(self,X):
        x_qs = self.x_qs.to(X.device) if torch.is_tensor(self.x_qs) else torch.from_numpy(self.x_qs).float().to(X.device)
        qs = self.qs.to(X.device) if torch.is_tensor(self.qs) else torch.from_numpy(self.qs).float().to(X.device)
        kappa = self.kappa.to(X.device) if torch.is_tensor(self.kappa) else self.kappa
        
        r_vec_expanded = X.unsqueeze(1)
        x_qs_expanded = x_qs.unsqueeze(0)
        r_diff = r_vec_expanded - x_qs_expanded
        r = torch.sqrt(torch.sum(r_diff**2, dim=2))
        q_over_r = qs*torch.exp(-kappa*r) / r
        total_sum = torch.sum(q_over_r, dim=1)
        result = (1 / (self.epsilon_2 * 4 * self.pi)) * total_sum
        result = result.unsqueeze(1)
        return result


    def G_L(self,X,Xp):
        Xp = Xp.to(X.device) if torch.is_tensor(Xp) and Xp.device != X.device else Xp
        X_expanded = X.unsqueeze(1)  # Shape: (n, 1, 3)
        Xp_expanded = Xp.unsqueeze(0)  # Shape: (1, m, 3)
        r_diff = X_expanded - Xp_expanded  # Shape: (n, m, 3)
        r = torch.sqrt(torch.sum(r_diff**2, dim=2))  # Shape: (n, m)
        over_r = 1 / r  # Shape: (n, m)
        green_function = (1 / (4 * self.pi)) * over_r  # Shape: (n, m)
        return green_function
    
    def G_Y(self,X,Xp):
        Xp = Xp.to(X.device) if torch.is_tensor(Xp) and Xp.device != X.device else Xp
        kappa = self.kappa.to(X.device) if torch.is_tensor(self.kappa) else self.kappa
        X_expanded = X.unsqueeze(1)  # Shape: (n, 1, 3)
        Xp_expanded = Xp.unsqueeze(0)  # Shape: (1, m, 3)
        r_diff = X_expanded - Xp_expanded  # Shape: (n, m, 3)
        r = torch.sqrt(torch.sum(r_diff**2, dim=2))  # Shape: (n, m)
        e_over_r = torch.exp(-kappa*r) / r  # Shape: (n, m)
        green_function = (1 / (4 * self.pi)) * e_over_r  # Shape: (n, m)
        return green_function
    
    def dG_L(self,X,Xp,N):
        Xp = Xp.to(X.device) if torch.is_tensor(Xp) and Xp.device != X.device else Xp
        N = N.to(X.device) if torch.is_tensor(N) and N.device != X.device else N
        X_expanded = X.unsqueeze(1)  # Shape: (n, 1, 3)
        Xp_expanded = Xp.unsqueeze(0)  # Shape: (1, m, 3)
        n_expanded = N.unsqueeze(0)  # Shape: (1, m, 3)
        r_diff = X_expanded - Xp_expanded  # Shape: (n, m, 3)
        r = torch.sqrt(torch.sum(r_diff**2, dim=2))  # Shape: (n, m)
        dg_dr = (-1 / (4 * self.pi)) / (r**2)  # Shape: (n, m)
        grad_x = -1*dg_dr * r_diff[:, :, 0] / r  # Shape: (n, m)
        grad_y = -1*dg_dr * r_diff[:, :, 1] / r  # Shape: (n, m)
        grad_z = -1*dg_dr * r_diff[:, :, 2] / r  # Shape: (n, m)
        grad = torch.stack([grad_x, grad_y, grad_z], dim=2)  # Shape: (n, m, 3)
        dg_dn = torch.sum(grad * n_expanded, dim=2)  # Shape: (n, m)
        return dg_dn

    def dG_Y(self,X,Xp,N):
        Xp = Xp.to(X.device) if torch.is_tensor(Xp) and Xp.device != X.device else Xp
        N = N.to(X.device) if torch.is_tensor(N) and N.device != X.device else N
        kappa = self.kappa.to(X.device) if torch.is_tensor(self.kappa) else self.kappa
        X_expanded = X.unsqueeze(1)  # Shape: (n, 1, 3)
        Xp_expanded = Xp.unsqueeze(0)  # Shape: (1, m, 3)
        n_expanded = N.unsqueeze(0)  # Shape: (1, m, 3)
        r_diff = X_expanded - Xp_expanded  # Shape: (n, m, 3)
        r = torch.sqrt(torch.sum(r_diff**2, dim=2))  # Shape: (n, m)
        dg_dr = (-torch.exp(-kappa*r) / (4 * self.pi)) / (r)  *(-kappa/r - 1/r**2) # Shape: (n, m)
        grad_x = -1*dg_dr * r_diff[:, :, 0] / r  # Shape: (n, m)
        grad_y = -1*dg_dr * r_diff[:, :, 1] / r  # Shape: (n, m)
        grad_z = -1*dg_dr * r_diff[:, :, 2] / r  # Shape: (n, m)
        grad = torch.stack([grad_x, grad_y, grad_z], dim=2)  # Shape: (n, m, 3)
        dg_dn = torch.sum(grad * n_expanded, dim=2)  # Shape: (n, m)
        return dg_dn
        

    @adim_function
    def charges_Born_Ion(self,r, R, q):
        if R is None:
            R = self.mesh.R_mol
        epsilon_1 = self.epsilon_1
        epsilon_2 = self.epsilon_2
        kappa = self.kappa

        return (q / (4 * self.pi)) * (-1 / (epsilon_1 * R) + 1 / (epsilon_2 * (1 + kappa * R) * R))



    @adim_function
    def analytic_Born_Ion(self,r, R=None, index_q=0):
        if R is None:
            R = self.mesh.R_mol
        epsilon_1 = self.epsilon_1
        epsilon_2 = self.epsilon_2
        kappa = self.kappa
        q = self.qs[index_q]

        f_IN = lambda r: (q/(4*self.pi)) * ( - 1/(epsilon_1*R) + 1/(epsilon_2*(1+kappa*R)*R) )
        f_OUT = lambda r: (q/(4*self.pi)) * (np.exp(-kappa*(r-R))/(epsilon_2*(1+kappa*R)*r) - 1/(epsilon_1*r))

        y = np.piecewise(r, [r<=R, r>R], [f_IN, f_OUT])

        return y 



    @adim_function
    def analytic_Born_Ion_du(self,r):
        rI = self.mesh.R_mol
        epsilon_1 = self.epsilon_1
        epsilon_2 = self.epsilon_2
        kappa = self.kappa
        q = self.q_list[0].q

        f_IN = lambda r: (q/(4*self.pi)) * ( -1/(epsilon_1*r**2) )
        f_OUT = lambda r: (q/(4*self.pi)) * (np.exp(-kappa*(r-rI))/(epsilon_2*(1+kappa*rI))) * (-kappa/r - 1/r**2)

        y = np.piecewise(r, [r<=rI, r>rI], [f_IN, f_OUT])

        return y

    @adim_function
    def Spherical_Harmonics(self, x, flag, R=None, N=20):

        q = self.qs.numpy()
        xq = self.x_qs.numpy()
        E_1 = self.epsilon_1.numpy()
        E_2 = self.epsilon_2.numpy()
        kappa = self.kappa.numpy()
        if R is None:
            R = self.mesh.R_max_dist

        PHI = np.zeros(len(x))

        for K in range(len(x)):
            rho = np.sqrt(np.sum(x[K,:] ** 2))
            zenit = np.arccos(x[K, 2] / rho)
            azim = np.arctan2(x[K, 1], x[K, 0])

            phi = 0.0 + 0.0 * 1j

            for n in range(N):
                for m in range(-n, n + 1):

                    Enm = 0.0
                    for k in range(len(q)):
                        rho_k = np.sqrt(np.sum(xq[k,:] ** 2))
                        zenit_k = np.arccos(xq[k, 2] / rho_k)
                        azim_k = np.arctan2(xq[k, 1], xq[k, 0])

                        Enm += (
                            q[k]
                            * rho_k**n
                            *4*np.pi/(2*n+1)
                            * sp.sph_harm(m, n, -azim_k, zenit_k)
                        )

                    Anm = Enm * (1/(4*np.pi)) * ((2*n+1)) / (np.exp(-kappa*R)* ((E_1-E_2)*n*self.get_K(kappa*R,n)+E_2*(2*n+1)*self.get_K(kappa*R,n+1)))
                    Bnm = 1/(R**(2*n+1))*(np.exp(-kappa*R)*self.get_K(kappa*R,n)*Anm - 1/(4*np.pi*E_1)*Enm)
                    
                    if flag=='molecule':
                        phi += Bnm * rho**n * sp.sph_harm(m, n, azim, zenit)
                    if flag=='solvent':
                        phi += Anm * rho**(-n-1)* np.exp(-kappa*rho) * self.get_K(kappa*rho,n) * sp.sph_harm(m, n, azim, zenit)

            PHI[K] = np.real(phi)
        
        torch_dtype = torch.float32 if self.DTYPE == 'float32' else torch.float64
        return torch.from_numpy(PHI).to(torch_dtype) 

    @staticmethod
    def get_K(x, n):
        K = 0.0
        n_fact = sp.factorial(n)
        n_fact2 = sp.factorial(2 * n)
        for s in range(n + 1):
            K += (
                2**s
                * n_fact
                * sp.factorial(2 * n - s)
                / (sp.factorial(s) * n_fact2 * sp.factorial(n - s))
                * x**s
            )
        return K

    
    def pbj_solution(self,X,flag):

        if not self.pbj_created:

            from .pbj_utils.pbj_interface import PBJ
            if hasattr(self, 'mesh'):
                self.pbj_mesh_density = self.mesh.density_mol
                self.pbj_mesh_generator = self.mesh.mesh_generator

            self.pbj_obj = PBJ(self.domain_properties,self.pqr_path,self.pbj_mesh_density,self.pbj_mesh_generator,self.results_path)
            self.pbj_phi = self.pbj_obj.simulation.solutes[0].results['phi'].coefficients.reshape(-1,1)
            self.pbj_vertices = np.array(self.pbj_obj.simulation.solutes[0].mesh.vertices).transpose()
            self.pbj_elements = np.array(self.pbj_obj.simulation.solutes[0].mesh.elements).transpose()
            self.pbj_created = True

        phi = self.pbj_obj.calculate_potential(X,flag)
        return phi*self.qe/(self.eps0 * self.ang_to_m*self.to_V)

    def apbs_solution(self,X):

        if not self.apbs_created:
            from .apbs_utils.apbs_interface import APBS
            
            self.apbs_obj = APBS(self.domain_properties,self.equation,self.pqr_path,self.results_path)
            self.apbs_created = True

        phi = self.apbs_obj.calculate_potential(X)
        return phi*self.kb*self.T/(self.qe*self.to_V)

