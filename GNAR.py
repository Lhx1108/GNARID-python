import numpy as np
from scipy import sparse

from sklearn.linear_model import Lasso

class VARLasso:
    def __init__(self, p=1, alpha=1):
        self.p = p
        self.alpha = alpha

    def fit(self,vts, max_iter=100000):
        N = vts.shape[1]
        X = np.hstack([vts[self.p-i:-i] for i in range(1,1+self.p)])
        Y = vts[self.p:]
        if self.alpha == 0:
            self.W = np.linalg.lstsq(X,Y, rcond=None)[0]
        else:
            self.W = np.zeros((N*self.p,N))
            for j in range(N):
                self.W[:,j] = Lasso(self.alpha,max_iter=max_iter,fit_intercept=False).fit(X,Y[:,j]).coef_

    def predict(self,vts):
        X = np.hstack([vts[self.p-1:]]+[vts[self.p-i:-i+1] for i in range(2,1+self.p)])
        return X@self.W

    def validate(self, vts):
        X = np.hstack([vts[self.p-i:-i] for i in range(1,1+self.p)])
        Y = vts[self.p:]
        Y_pred = X@self.W
        return np.sum((Y_pred-Y)**2)
    

def zero_expand_matrix(A):
    T, N = A.shape
    
    if not isinstance(A, sparse.csr_matrix):
        A = sparse.csr_matrix(A)  # Convert to CSR if not already sparse
    
    row_indices, col_indices = A.nonzero()  # Get the indices of nonzero elements
    new_row_indices = row_indices + col_indices * T  # Adjust row indices for expansion
    data = A.data  # Keep the nonzero values

    # Create the sparse output matrix
    B = sparse.coo_matrix((data, (new_row_indices, col_indices)), shape=(T * N, N))

    return B.tocsr()  # Convert to CSR for efficient row operations


class NetworkModel:
    def __init__(self, alpha_order, beta_order, intercept=True, global_intercept=False, global_alpha=True, global_beta=True):
        self.beta_order = beta_order
        self.intercept = intercept
        self.global_intercept = global_intercept
        self.global_alpha = global_alpha
        self.global_beta = global_beta
        if isinstance(alpha_order,int):
            self.alpha_order = alpha_order
            self.alpha_orders = [1 for _ in range(alpha_order)]
        else:
            self.alpha_order = len(alpha_order)
            self.alpha_orders = alpha_order # alpha_orders is [1,0,0,1,...]; alpha order is max time lag

    def transformVTS(self,vts):
        X = []
        for t in range(self.alpha_order):
            if self.beta_order[t] != 0:
                X_t_beta = np.transpose(vts[np.newaxis,self.alpha_order-t-1:-t-1,:]@self.network.w_mats[1:1+self.beta_order[t]],axes=(1,2,0))
                if self.global_beta:
                    X_t_beta = X_t_beta.reshape((-1,self.beta_order[t]),order="F")
                    if not self.global_alpha:
                        X_t_beta = sparse.csr_matrix(X_t_beta)
                else:
                    X_t_beta = sparse.hstack([zero_expand_matrix(X_t_beta[:,:,i]) for i in range(self.beta_order[t])])
            else:
                X_t_beta = np.empty(((vts.shape[0]-self.alpha_order)*vts.shape[1],0))
                if (not self.global_alpha) or (not self.global_beta):
                    X_t_beta = sparse.csr_matrix(X_t_beta)
            if self.alpha_orders[t] == 1:
                X_t_alpha =vts[self.alpha_order-t-1:-t-1,:]
                if self.global_alpha:
                    X_t_alpha = X_t_alpha.reshape((-1,1),order="F")
                    if not self.global_beta:
                        X_t_alpha = sparse.csr_matrix(X_t_alpha)
                else:
                    X_t_alpha = zero_expand_matrix(X_t_alpha)
            else:
                X_t_alpha = np.empty(((vts.shape[0]-self.alpha_order)*vts.shape[1],0))
                if (not self.global_alpha) or (not self.global_beta):
                    X_t_alpha = sparse.csr_matrix(X_t_alpha)
            
            X.append(self.hstack((X_t_alpha,X_t_beta)))
        
        if self.intercept:
            if self.global_intercept:
                X_intercept = np.ones((self.network.size*(len(vts)-self.alpha_order),1))
            else:
                X_intercept = np.repeat(np.identity(self.network.size), len(vts)-self.alpha_order, axis=0)
            if not (self.global_alpha and self.global_beta):
                X_intercept = sparse.csr_matrix(X_intercept)
            X.append(X_intercept)
        X = self.hstack(X)
        return X

    def hstack(self,x):
        if self.global_alpha and self.global_beta:
            return np.hstack(x)
        else:
            return sparse.hstack(x)
    
    def return_coef(self, t, type="all"):
        if type=="alpha":
            if self.coef_index[2*t-2] != 0 and self.coef_index[2*t-2] != self.coef_index[2*t-3]:
                return self.coefs[self.coef_index[2*t-2]-1]
            else:
                return 0
        elif type == "beta":
            return self.coefs[self.coef_index[2*t-2]:self.coef_index[2*t-1]]
        elif type == "all":
            return self.coefs[self.coef_index[2*t-2]-1:self.coef_index[2*t-1]]

    def AIC(self):
        return 2*len(self.coefs)-2*self.loglik()
    def BIC(self):
        return len(self.coefs)*np.log(len(self.y))-2*self.loglik()
    def group_NTS(self, network, vts, size):
        self.network = network
        X = self.transformVTS(vts)
        T,n = len(vts),self.network.size
        self.k = int(np.ceil((T-self.alpha_order)/size))
        G = np.zeros(((T-self.alpha_order)*n,self.k*X.shape[1]))
        for i in range(self.k-1):
            for j in range(i+1):
                G[i*size*n:(i+1)*size*n,j*X.shape[1]:(j+1)*X.shape[1]] = X[i*size*n:(i+1)*size*n,:] 
        for j in range(self.k):
            G[(self.k-1)*size*n:,j*X.shape[1]:(j+1)*X.shape[1]] = X[(self.k-1)*size*n:,:]
        return G

from scipy.optimize import lsq_linear
from scipy.stats import multivariate_normal, norm
from itertools import chain, combinations


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def min_part(xs, r):
    xs = np.sort(xs)
    part = []
    ys = []
    for x in xs:
        for y in ys:
            if abs(x-y) >= r:
                part.append(ys)
                ys = []
                break
        ys.append(x)
    part.append(ys)
    return part
            

class GNAR(NetworkModel):
    def __init__(self, alpha_order, beta_order, intercept=True, global_intercept=False, global_alpha=True, global_beta=True):
        super().__init__(alpha_order, beta_order,intercept,global_intercept, global_alpha, global_beta)
        self.coef_index = np.cumsum([[self.alpha_orders[i]]+[beta_order[i]] for i in range(self.alpha_order)])
        self.coef_order = self.coef_index[-1]
        
    def simulate(self, network, initial_vts, length, coefs, error_cov_mat):
        self.network = network
        l = len(initial_vts)
        vts_sim = np.zeros((length+l,self.network.size))
        vts_sim[:l,:] = initial_vts
        for i in range(length):
            vts_sim[l+i,:] = self.transformVTS(vts_sim[l+i-self.alpha_order:l+i+1,:])@coefs + multivariate_normal(
                cov=error_cov_mat).rvs(1)
        return vts_sim[len(initial_vts):]
    
    def loglik(self):
        return np.sum(norm.logpdf(self.y,loc=self.X@self.coefs,scale=self.sigma2**.5))
    
    def r2(self):
        return r2_score(self.y, self.X@self.coefs)

    def adj_r2(self):
        return 1-((1-self.r2())*(len(self.y)-1)/(len(self.y)-len(self.coefs)-1))
        
    def fit(self, network, vts, use_ls=True, l2_penal=0):
        # the order of cols of vts must match the order of network nodes
        # i.e. the first vts col is the ts for the first node in the network, etc
        self.network = network
        self.vts = vts
        self.vts_end = vts[-self.alpha_order:,:]
        self.y = vts[self.alpha_order:,:].flatten("F")
        self.X = self.transformVTS(vts)
        if self.global_alpha:
            if l2_penal != 0:
                self.coefs = np.linalg.solve(self.X.T@self.X + l2_penal*np.identity(self.X.shape[1]), self.X.T@self.y)
            else:
                if use_ls:
                    self.coefs,self.res = np.linalg.lstsq(self.X, self.y, rcond=None)[:2]
                else:
                    self.coefs = np.linalg.solve(self.X.T@self.X, self.X.T@self.y)
        else:
            if use_ls:
                self.coefs = sparse.linalg.lsqr(self.X,self.y)[0]
            else:
                self.coefs = sparse.linalg.spsolve(self.X.T@self.X, self.X.T@self.y)
        self.vts_fitted = np.reshape((self.X@self.coefs),(-1,network.size),"F")
        #self.sigma2 = np.sum((self.y-self.X@self.coefs)**2)/(len(self.y)-len(self.coefs)) #assuming equal error var
        
    def compute_cov(self,X,y,coefs):
        T = int(len(y)/self.network.size)
        residual = np.reshape((y-X@coefs),(-1,self.network.size),"F")
        res_cov_mat = residual.T@residual/T
        var_X_res = 0
        X = X[np.concatenate([np.array([i*T for i in range(self.network.size)])+t for t in range(T)])]
        y = y[np.concatenate([np.array([i*T for i in range(self.network.size)])+t for t in range(T)])]
        for i in range(self.network.size):
            for j in range(self.network.size):
                xti_xtj = np.mean([np.outer(X[i+k*self.network.size],X[j+k*self.network.size]) for k in range(T)],axis=0)
                var_X_res += res_cov_mat[i,j]*xti_xtj
        X_cov_mat = X.T@X/T
        inv_X_cov_mat = np.linalg.inv(X.T@X)*T
        coef_cov_mat = inv_X_cov_mat@var_X_res@inv_X_cov_mat/T
        inv_coef_cov_mat = X_cov_mat@np.linalg.solve(var_X_res,X_cov_mat)*T
        return coef_cov_mat,inv_coef_cov_mat
        
    def predict(self, length, nodes=None, vts_end =None):
        if vts_end is None:
            vts_end = self.vts_end
        vts_pred = np.zeros((length+self.alpha_order,self.network.size))
        vts_pred[:self.alpha_order,:] = vts_end
        for i in range(length):
            vts_pred[self.alpha_order+i,:] = self.transformVTS(vts_pred[i:self.alpha_order+i+1,:])@self.coefs
        if nodes is None:
            return vts_pred[self.alpha_order:,:]
        else:
            return vts_pred[self.alpha_order:,nodes]


    
    def return_coef(self, t, type="all"):
        if type=="alpha":
            if self.coef_index[2*t-2] != 0 and self.coef_index[2*t-2] != self.coef_index[2*t-3]:
                return self.coefs[self.coef_index[2*t-2]-1]
            else:
                return 0
        elif type == "beta":
            return self.coefs[self.coef_index[2*t-2]:self.coef_index[2*t-1]]
        elif type == "all":
            return self.coefs[self.coef_index[2*t-2]-1:self.coef_index[2*t-1]]
        
    def initial_fit(self,network,vts,size,level=0.05):
        self.network = network
        self.vts = vts
        self.size = size
        self.T = len(self.vts)-self.alpha_order
        self.y = vts[self.alpha_order:,:].flatten("F")
        self.X = self.transformVTS(vts)
        self.block_number = int(np.ceil(self.T/size))
        index = np.concatenate([np.arange(size)+i*self.T for i in range(self.network.size)])
        coefs_prev = lsq_linear(self.X[index], self.y[index]).x
        self.cpts_candidate = []
        for l in range(1,self.block_number):
            index = np.concatenate([np.arange(l*size,min(self.T,(l+1)*size))+i*self.T for i in range(self.network.size)])
            coefs_new = lsq_linear(self.X[index], self.y[index]).x
            coef_cov_mat,inv_coef_cov_mat = self.compute_cov(self.X[index], self.y[index],coefs_new)
            test_chi2 = np.dot((coefs_new-coefs_prev),(inv_coef_cov_mat@(coefs_new-coefs_prev)))
            p = chi2(df=len(coefs_prev)).sf(test_chi2)
            #print(p)
            #print(coef_cov_mat)
            #print(2*norm().sf(abs(coefs_new-coefs_prev)/np.array([coef_cov_mat[i,i] for i in range(len(coefs_new))])**(1/2)))
            if p<level:
                coefs_prev = coefs_new
                self.cpts_candidate.append(l*size)
        #print(self.cpts_candidate)
            
    def minimize(self,X,y):
        return lsq_linear(X,y).cost
            
    def LIC(self,cpts_subset,window,omega):
        #local information criterion
        rss = 0
        n = 0
        for cpt in self.cpts_candidate:
            if (cpt in cpts_subset):
                index_1 = np.concatenate([np.arange(cpt-window,cpt)+i*(self.T) for i in range(self.network.size)])
                index_2 = np.concatenate([np.arange(cpt,min(self.T,cpt+window))+i*(self.T) for i in range(self.network.size)])
                rss += (lsq_linear(self.X[index_1],self.y[index_1]).cost+
                lsq_linear(self.X[index_2],self.y[index_2]).cost)
                n += len(index_1) + len(index_2)
            else:
                index = np.concatenate([np.arange(cpt-window,min(self.T,cpt+window))+i*(self.T) for i in range(self.network.size)])
                rss += lsq_linear(self.X[index],self.y[index]).cost
                n += len(index)
        if omega == "AIC":
            lic = n*np.log(rss/n) + len(cpts_subset)*self.X.shape[1]*2
        elif omega == "BIC":
            lic = n*np.log(rss/n) + len(cpts_subset)*self.X.shape[1]*np.log(n)
        else:
            lic = n*np.log(rss/n) + len(cpts_subset)*self.X.shape[1]*omega
        return lic
            
    def local_screening(self,window,omega):
        lowest_lic = np.inf
        for cpts_subset in powerset(self.cpts_candidate):
            lic = self.LIC(cpts_subset,window,omega)
            if lic < lowest_lic:
                cpts = cpts_subset
                lowest_lic = lic
        self.cpts = cpts
    
    
    def exhaustive_search(self,window):
        partition = min_part(self.cpts,2*window)
        cpts = []
        for ts in partition:
            if len(ts) == 1:
                l,u = ts[0]-window, min(self.T,ts[0]+window)
            else:
                l,u = min(ts),max(ts)
            lowest_cost = np.inf
            for t in range(l,u):
                index_1 = np.concatenate([np.arange(l,t)+i*(len(self.vts)-self.alpha_order) for i in range(self.network.size)])
                index_2 = np.concatenate([np.arange(t,u)+i*(len(self.vts)-self.alpha_order) for i in range(self.network.size)])
                cost = lsq_linear(self.X[index_1],self.y[index_1]).cost + lsq_linear(self.X[index_2],self.y[index_2]).cost
                if cost < lowest_cost:
                    best_t = t
                    lowest_cost = cost
            cpts.append(best_t)
        self.cpts = np.array(cpts)+self.alpha_order
        
    def cpts_detect(self,network,vts,size,window,omega):
        self.initial_fit(network,vts,size)
        if len(self.cpts_candidate) == 0:
            return []
        self.local_screening(window, omega)
        if len(self.cpts) == 0:
            return []
        self.exhaustive_search(window)
        return self.cpts

    def validate(self, vts):
        X = self.transformVTS(vts)
        y = vts[self.alpha_order:,:].flatten("F")
        return np.sum((y-X@self.coefs)**2)
            

def GNAR_sim_piecewise(networks, alpha_order, beta_order, coefs_list, intercept,global_intercept,error_cov_mat_list, length_list, burn_in = 100):
    num_piece = len(coefs_list)
    vts = np.zeros((sum(length_list)+burn_in+alpha_order,networks[0].size))
    vts[:alpha_order] = norm(loc=0).rvs(size=(alpha_order,networks[0].size))
    current_index = alpha_order
    length_list[0] += burn_in
    for l in range(num_piece):
        model = GNAR(alpha_order,beta_order,intercept, global_intercept)
        vts[current_index:current_index+length_list[l]]=model.simulate(networks[l],vts[:current_index],length_list[l],coefs_list[l],error_cov_mat_list[l])
        current_index = current_index+length_list[l]
    return vts[burn_in+alpha_order:]

def standardize(arr):
    """
    Standardize a 2D numpy array column-wise.
    Each column will have mean 0 and standard deviation 1.
    
    Parameters:
        arr (np.ndarray): 2D array to standardize.
        
    Returns:
        np.ndarray: Standardized 2D array.
    """
    arr = np.array(arr, dtype=float)  # ensure float for division
    means = np.mean(arr, axis=0)      # column means
    stds = np.std(arr, axis=0, ddof=0)  # column std deviations
    
    # Avoid division by zero
    stds[stds == 0] = 1
    
    standardized = (arr - means) / stds
    return standardized
            