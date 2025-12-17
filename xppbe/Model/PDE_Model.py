import os
import numpy as np
import torch

from xppbe.Mesh.Charges_utils import get_charges_list
from .Solutions_utils import Solution_utils

class PBE(Solution_utils):

    DTYPE = 'float32'
    dtype = torch.float32

    qe = torch.tensor(1.60217663e-19, dtype=dtype)
    eps0 = torch.tensor(8.8541878128e-12, dtype=dtype)     
    kb = torch.tensor(1.380649e-23, dtype=dtype)              
    Na = torch.tensor(6.02214076e23, dtype=dtype)
    ang_to_m = torch.tensor(1e-10, dtype=dtype)
    cal2j = torch.tensor(4.184, dtype=dtype)

    pi = torch.tensor(np.pi, dtype=dtype)

    def __init__(self, domain_properties, mesh, equation, pinns_method, adim, main_path, molecule_dir, results_path):      

        self.mesh = mesh
        self.main_path = main_path
        self.equation = equation
        self.pinns_method = pinns_method
        self.adim = adim
        self.molecule_path = molecule_dir
        self.results_path = results_path

        self.calculate_properties(domain_properties)

        self.get_charges()
        if self.scheme == 'direct':
            self.bempp = None
            self.get_integral_operators()

        super().__init__()

    @property
    def get_PDEs(self):
        PDEs = [self.PDE_in,self.PDE_out]
        return PDEs
    
    def calculate_properties(self,domain_properties):

        self.domain_properties = {
                'molecule': 'born_ion',
                'epsilon_1':  1,
                'epsilon_2': 80,
                'kappa': 0.125,
                'T' : 300 
                }
        
        T = domain_properties['T'] if 'T' in domain_properties else self.domain_properties['T']
        kappa = domain_properties['kappa'] if 'kappa' in domain_properties else self.domain_properties['kappa']
        epsilon_2 = domain_properties['epsilon_2'] if 'epsilon_2' in domain_properties else self.domain_properties['epsilon_2']

        domain_properties['concentration'] = (kappa/self.ang_to_m)**2*(self.eps0*epsilon_2*self.kb*T)/(2*self.qe**2*self.Na)/1000

        qe_eps0_ang = self.qe/(self.eps0 * self.ang_to_m)  
        fact = self.kb/(qe_eps0_ang*self.qe)
        if self.adim == 'qe_eps0_angs':
            self.to_V = self.qe/(self.eps0 * self.ang_to_m)  
            domain_properties['beta'] = 1
            domain_properties['gamma'] = T*fact
        elif self.adim == 'kb_T_qe':
            self.to_V = self.kb*T/self.qe
            domain_properties['beta'] = T*fact
            domain_properties['gamma'] = 1
        
        for key in ['molecule','epsilon_1','epsilon_2','kappa','T','concentration','beta','gamma']:
            if key in domain_properties:
                self.domain_properties[key] = domain_properties[key]
            if key != 'molecule':
                torch_dtype = torch.float32 if self.DTYPE == 'float32' else torch.float64
                value = self.domain_properties[key]
                if torch.is_tensor(value):
                    setattr(self, key, value.clone().detach().to(torch_dtype))
                else:
                    setattr(self, key, torch.tensor(value, dtype=torch_dtype))
            else:
                setattr(self, key, self.domain_properties[key])

        self.sigma = self.mesh.G_sigma
            
    def get_phi_interface(self,X,model,**kwargs):      
        u_mean = self.get_phi(X,'interface',model,**kwargs)
        u_1 = self.get_phi(X,'molecule',model,**kwargs)
        u_2 = self.get_phi(X,'solvent',model,**kwargs)
        return u_mean[:,0],u_1[:,0],u_2[:,0]
    
    def get_dphi_interface(self,X,N_v,model,value='phi'): 
        du_1,du_2 = self.get_dphi(X,N_v,'',model,value)
        du_prom = (du_1*self.PDE_in.epsilon + du_2*self.PDE_out.epsilon)/2
        return du_prom,du_1,du_2

    def get_phi_interface_verts(self,model,**kwargs):      
        verts = torch.from_numpy(self.mesh.mol_verts).to(torch.float32 if self.DTYPE == 'float32' else torch.float64)
        # Move to same device as model
        device = next(model.parameters()).device
        verts = verts.to(device)
        return self.get_phi_interface(verts,model,**kwargs)
    
    def get_dphi_interface_verts(self,model,value='phi'): 
        verts = torch.from_numpy(self.mesh.mol_verts).to(torch.float32 if self.DTYPE == 'float32' else torch.float64)
        # Move to same device as model
        device = next(model.parameters()).device
        verts = verts.to(device)     
        N_v = torch.from_numpy(self.mesh.mol_verts_normal).to(torch.float32 if self.DTYPE == 'float32' else torch.float64).to(device)
        return self.get_dphi_interface(verts,N_v,model)
    
    
    def get_phi_ens(self,model,X_mesh,q_L, method='mean', pinn=True, known_method=False):        

        (X_solv,flag) = X_mesh
        phi_ens_L = list()

        for x_q,r_q in q_L:
            
            if method=='exponential':
                if pinn: 
                    phi = self.get_phi(X_solv,flag, model)
                else:
                    X_solv_torch = torch.from_numpy(X_solv).float() if isinstance(X_solv, np.ndarray) else X_solv
                    phi = self.phi_known(known_method,'phi',X_solv_torch,'solvent').reshape(-1,1)
                r_H = torch.sqrt(torch.sum((x_q - X_solv)**2, dim=1, keepdim=True))
                G2_p = torch.sum(self.aprox_exp(-phi/self.gamma)/r_H**6)
                G2_m = torch.sum(self.aprox_exp(phi/self.gamma)/r_H**6)
                phi_ens_pred = - self.gamma/2 * torch.log(G2_p/G2_m)
            
            elif method=='mean':
                r_H = torch.sqrt(torch.sum((x_q - X_solv)**2, dim=1))
                mask = r_H < (r_q + self.mesh.dR_exterior)
                X_ens = X_solv[mask]
                if pinn: 
                    phi = self.get_phi(X_ens,flag, model)
                else:
                    X_ens_torch = torch.from_numpy(X_ens).float() if isinstance(X_ens, np.ndarray) else X_ens
                    phi = self.phi_known(known_method,'phi',X_ens_torch,'solvent').reshape(-1,1)
                phi_ens_pred = torch.mean(phi)

            phi_ens_L.append(phi_ens_pred)

        return phi_ens_L    
    
    def solvation_energy_phi_qs(self,phi_q):
        # Convert to torch if needed
        if isinstance(phi_q, np.ndarray):
            phi_q = torch.from_numpy(phi_q).float()
        qs = torch.from_numpy(self.qs).float() if isinstance(self.qs, np.ndarray) else self.qs
        G_solv = 0.5*torch.sum(qs * phi_q)
        G_solv *= self.to_V*self.qe*self.Na*(10**-3/self.cal2j)   
        return G_solv

    # Losses

    def get_loss(self, X_batches, model, validation=False):
        # Get device from model parameters
        device = next(model.parameters()).device
        L = self.create_L(device=device)

        #residual
        if 'R1' in X_batches: 
            ((X,SU),flag) = X_batches['R1']
            loss_r = self.PDE_in.residual_loss(self.mesh,model,self.mesh.get_X(X),SU,flag)
            L['R1'] += loss_r   

        if 'R2' in X_batches: 
            ((X,SU),flag) = X_batches['R2']
            loss_r = self.PDE_out.residual_loss(self.mesh,model,self.mesh.get_X(X),SU,flag)
            L['R2'] += loss_r   

        if 'Q1' in X_batches: 
            ((X,SU),flag) = X_batches['Q1']
            loss_q = self.PDE_in.residual_loss(self.mesh,model,self.mesh.get_X(X),SU,flag)
            L['Q1'] += loss_q 

        #dirichlet 
        if 'D2' in X_batches:
            ((X,U),flag) = X_batches['D2']
            loss_d = self.dirichlet_loss(self.mesh,model,X,U,flag)
            L['D2'] += loss_d

        # data known
        if 'K1' in X_batches and not validation:
            ((X,U),flag) = X_batches['K1']
            loss_k = self.dirichlet_loss(self.mesh,model,X,U,flag)
            L['K1'] += loss_k   

        if 'K2' in X_batches and not validation:
            ((X,U),flag) = X_batches['K2']
            loss_k = self.dirichlet_loss(self.mesh,model,X,U,flag)
            L['K2'] += loss_k 

        if 'I' in X_batches:
            if 'Iu' in self.mesh.domain_mesh_names:
                L['Iu'] += self.get_loss_I(model,X_batches['I'], 'Iu')
            if 'Id' in self.mesh.domain_mesh_names:
                L['Id'] += self.get_loss_I(model,X_batches['I'], 'Id')
            if 'Ir' in self.mesh.domain_mesh_names:
                L['Ir'] += self.get_loss_I(model,X_batches['I'], 'Ir')    

            if 'IB1' in self.mesh.domain_mesh_names: 
                ((X,N),flag) = X_batches['I']
                loss_r = self.PDE_in.residual_loss(self.mesh,model,self.mesh.get_X(X),N,flag)
                L['IB1'] += loss_r   

            if 'IB2' in self.mesh.domain_mesh_names: 
                ((X,N),flag) = X_batches['I']
                loss_r = self.PDE_out.residual_loss(self.mesh,model,self.mesh.get_X(X),N,flag)
                L['IB2'] += loss_r   

        if 'E2' in X_batches and not validation:
            L['E2'] += self.get_loss_experimental(model,X_batches['E2'])

        if 'G' in X_batches and not validation:
            L['G'] += self.get_loss_Gauss(model,X_batches['G'])

        return L


    def dirichlet_loss(self,mesh,model,XD,UD,flag):
        Loss_d = 0
        u_pred = self.get_phi(XD,flag,model)
        loss = torch.mean((UD - u_pred)**2)
        Loss_d += loss
        return Loss_d

    def get_loss_I(self,model,XI_data,loss_type='Iu'):
        
        loss = 0
        ((XI,N_v),flag) = XI_data
        X = self.mesh.get_X(XI)

        if loss_type=='Iu':
            u1 = self.get_phi(XI,'molecule',model)
            u2 = self.get_phi(XI,'solvent',model)
            loss += torch.mean((u1-u2)**2)

        elif loss_type=='Id':
            du_1,du_2 = self.get_dphi(XI,N_v,flag,model,value='phi')
            loss += torch.mean((du_1*self.PDE_in.epsilon - du_2*self.PDE_out.epsilon)**2)
        
        elif loss_type=='Ir':
            r1 = self.PDE_in.get_r(self.mesh,model,X,None,'molecule')
            r2 = self.PDE_out.get_r(self.mesh,model,X,None,'solvent')
            loss += torch.mean((r1-r2)**2)
            
        return loss

    def get_loss_experimental(self,model,X_exp):             

        loss = torch.tensor(0.0, dtype=self.dtype)
        n = len(X_exp)
        ((X,X_values),flag,method) = X_exp
        q_L,phi_ens_exp_L = zip(*X_values)
        phi_ens_pred_L = self.get_phi_ens(model,(X,flag),q_L,method)

        for phi_pred,phi_exp in zip(phi_ens_pred_L,phi_ens_exp_L):
            loss += (phi_pred - phi_exp)**2

        loss *= (1/n)

        return loss
    
    def get_loss_Gauss(self,model,XI_data):
        loss = 0
        ((XI,N_v,areas),flag) = XI_data
        du_1,du_2 = self.get_dphi(XI,N_v,flag,model,value='phi')
        du_prom = (du_1*self.PDE_in.epsilon + du_2*self.PDE_out.epsilon)/2

        integral = torch.sum(du_prom * areas)
        loss += torch.mean((integral - self.total_charge)**2)

        return loss


    @staticmethod
    def aprox_exp(x):
        aprox = 1.0 + x + x**2/2.0 + x**3/6.0 + x**4/24.0
        return aprox
    
    @staticmethod
    def aprox_sinh(x):
        aprox = x + x**3/6.0 
        return aprox

    # Differential operators

    def laplacian(self,mesh,model,X,flag,value='phi'):
        x,y,z = X
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)
        z = z.requires_grad_(True)
        
        R = mesh.stack_X(x,y,z)
        u = self.get_phi(R,flag,model,value)
        
        # First derivatives
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_z = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        
        # Second derivatives
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        u_zz = torch.autograd.grad(u_z, z, grad_outputs=torch.ones_like(u_z), create_graph=True)[0]
        
        return u_xx + u_yy + u_zz

    def gradient(self,mesh,model,X,flag,value='phi'):
        x,y,z = X
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)
        z = z.requires_grad_(True)
        
        R = mesh.stack_X(x,y,z)
        u = self.get_phi(R,flag,model,value)
        
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_z = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        
        return (u_x,u_y,u_z)
    
    def directional_gradient(self,mesh,model,X,n_v,flag,value='phi'):
        gradient = self.gradient(mesh,model,X,flag,value)
        # Ensure n_v tensors are on the same device as gradients
        n_v = [nv.to(gradient[0].device) if torch.is_tensor(nv) else nv for nv in n_v]
        dir_deriv = 0
        for j in range(3):
            dir_deriv += n_v[j]*gradient[j]
        return dir_deriv
    
    # utils

    def get_charges(self):
        self.pqr_path = os.path.join(self.molecule_path,self.molecule+'.pqr')
        self.q_list = get_charges_list(self.pqr_path)

        n = len(self.q_list)
        self.qs = np.zeros(n)
        self.x_qs = np.zeros((n,3))
        for i,q in enumerate(self.q_list):
            self.qs[i] = q.q
            self.x_qs[i,:] = q.x_q        
        torch_dtype = torch.float32 if self.DTYPE == 'float32' else torch.float64
        self.total_charge = torch.tensor(np.sum(self.qs), dtype=torch_dtype)
        self.qs = torch.from_numpy(self.qs).to(torch_dtype)
        self.x_qs = torch.from_numpy(self.x_qs).to(torch_dtype)

        radii = np.array([q.r_q for q in self.q_list])
        radii[radii<1e-6] = np.mean(radii)

        scale_min_value_1, scale_max_value_1 = 0., 0.
        scale_min_value_2, scale_max_value_2 = 0., 0.

        positions = torch.from_numpy(self.x_qs).float() if isinstance(self.x_qs, np.ndarray) else self.x_qs
        radii = np.array([q.r_q for q in self.q_list])
        radii[radii<1e-6] = np.mean(radii)
        radii = torch.from_numpy(radii).float()
        num_charges = len(self.q_list)

        positions_expanded = positions.unsqueeze(1)
        positions_diff = torch.norm(positions_expanded - positions, dim=2)
        mask = positions_diff != 0

        phi_born_all = self.charges_Born_Ion(0.0,radii,self.qs)

        phi_contribs_all = []
        for idx in range(num_charges):
            valid_indices = torch.where(mask[idx])[0]
            diff_positions = positions_diff[idx][mask[idx]]
            if len(valid_indices) > 0:
                phi_contribs = torch.stack([
                    self.charges_Born_Ion(diff_positions[j].item(), R=diff_positions[j].item(), q=self.qs[valid_indices[j].item()])
                    for j in range(len(valid_indices))
                ])
                phi_contribs_all.append(torch.sum(phi_contribs))
            else:
                phi_contribs_all.append(torch.tensor(0.0, dtype=self.dtype))
        phi_contribs_all = torch.stack(phi_contribs_all)

        phi_total_all = phi_born_all + phi_contribs_all

        radii_col = radii.unsqueeze(1)
        zeros_col = torch.zeros_like(positions[:, 1:3])
        G_additions = self.G(positions + torch.cat([radii_col, zeros_col], dim=1))

        phi_max_all = torch.maximum(phi_born_all, phi_total_all).reshape(-1,1)
        phi_min_all = torch.minimum(phi_born_all, phi_total_all).reshape(-1,1)

        phi_1_max_all = phi_max_all + G_additions if self.fields[0] == 'phi' else phi_max_all
        phi_1_min_all = phi_min_all + G_additions if self.fields[0] == 'phi' else phi_min_all
        phi_2_max_all = phi_max_all + G_additions if self.fields[1] == 'phi' else phi_max_all
        phi_2_min_all = phi_min_all + G_additions if self.fields[1] == 'phi' else phi_min_all

        phi_1_max = torch.max(phi_1_max_all)
        phi_1_min = torch.min(phi_1_min_all)
        phi_2_max = torch.max(phi_2_max_all)
        phi_2_min = torch.min(phi_2_min_all)

        scale_max_value_1 = torch.maximum(torch.tensor(0.0), phi_1_max)
        scale_min_value_1 = torch.minimum(torch.tensor(0.0), phi_1_min)
        scale_max_value_2 = torch.maximum(torch.tensor(0.0), phi_2_max)
        scale_min_value_2 = torch.minimum(torch.tensor(0.0), phi_2_min)

        self.scale_phi_1 = [float(scale_min_value_1), float(scale_max_value_1)]
        self.scale_phi_2 = [float(scale_min_value_2), float(scale_max_value_2)]


    def get_integral_operators(self):
        if self.bempp == None:
            import bempp.api
            self.bempp = bempp.api
        elements = self.mesh.mol_faces
        vertices = self.mesh.mol_verts
        self.grid = self.bempp.Grid(vertices.transpose(), elements.transpose())
        self.space = self.bempp.function_space(self.grid, "DP", 0)
        self.dirichl_space = self.space
        self.neumann_space = self.space

        self.slp_q = bempp.api.operators.potential.laplace.single_layer(self.neumann_space, self.x_qs.numpy().transpose())
        self.dlp_q = bempp.api.operators.potential.laplace.double_layer(self.dirichl_space, self.x_qs.numpy().transpose())

        vertices = self.grid.vertices
        faces_normals = self.grid.normals
        elements = self.grid.elements
        centroids = np.zeros((3, elements.shape[1]))
        for i, element in enumerate(elements.T):
            centroids[:, i] = np.mean(vertices[:, element], axis=1)

        torch_dtype = torch.float32 if self.DTYPE == 'float32' else torch.float64
        self.mesh.grid_centroids = torch.from_numpy(centroids.transpose()).to(torch_dtype).reshape(-1,3)
        self.mesh.grid_faces_normals = torch.from_numpy(faces_normals.transpose()).to(torch_dtype).reshape(-1,3)
    
    def get_grid_coefficients_faces(self,model):

        X = self.mesh.grid_centroids
        Nv = self.mesh.grid_faces_normals
        phi_mean,_,_ = self.get_phi_interface(X,model)
        u_interface = phi_mean.numpy().flatten()
        _,du_1,_ = self.get_dphi_interface(X,Nv,model)
        du_1_interface = du_1.numpy().flatten()

        phi = self.bempp.GridFunction(self.space, coefficients=u_interface)
        dphi = self.bempp.GridFunction(self.space, coefficients=du_1_interface)

        return phi,dphi


    @classmethod
    def create_L(cls, device=None):
        cls.names = ['R1','D1','N1','K1','Q1','R2','D2','N2','K2','G','Iu','Id','Ir','E2','P1','P2','IB1','IB2']
        L = dict()
        torch_dtype = torch.float32 if cls.DTYPE == 'float32' else torch.float64
        if device is None:
            device = torch.device('cpu')
        for t in cls.names:
            L[t] = torch.tensor(0.0, dtype=torch_dtype, device=device)
        return L



