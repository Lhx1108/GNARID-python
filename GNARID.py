import numpy as np
from scipy import sparse
from network import *
from GNAR import *
from copy import deepcopy


class GNARID(NetworkModel):
    def __init__(self, alpha_order, beta_order, intercept=True, global_intercept=False, global_alpha=True, global_beta=True):
        """
        GNAR model with inverse distance weights and joint optimisation over coefficients and gamma.
    
        Coefficients are estimated by least squares (or ridge), while `gamma` can be chosen either 
        by grid search or updated by gradient descent together with the coefficients.
    
        Parameters
        ----------
        alpha_order : int
            Autoregressive order (number of temporal lags).
        beta_order : list[int] or array-like
            Per-lag network stage orders. `beta_order[t]` is the number of network stages
            used at lag t (0-indexed).
        intercept : bool, default=True
            Whether to include an intercept term.
        global_intercept : bool, default=False
            If True, use a single global intercept; otherwise use node-specific intercepts.
        global_alpha : bool, default=True
            If True, use a global alpha for all alpha_i. Non-global options
            use sparse solvers.
        global_beta : bool, default=True
            If True, use a global beta_{j,r} for all beta_{j,r,i}. 
            (gradient descent currently assumes global alpha/beta).
    
        Attributes
        ----------
        coefs : ndarray
            Fitted coefficient vector, ordered to align with the design matrix.
        gamma : float
            Fitted distance exponent used to construct network weights.
        losses : list[float]
            Values of sum-of-squared residuals recorded during fitting
            (definition differs slightly between grid_search and gradient loop).
        gamma_hist : list[float]
            History of gamma values visited during gradient-based optimisation.
        """
        # gradient descent does not support non-global alpha/beta yet
        super().__init__(alpha_order, beta_order,intercept,global_intercept,global_alpha,global_beta)
        self.coef_index = np.cumsum([[self.alpha_orders[i]]+[beta_order[i]] for i in range(self.alpha_order)])
        self.coef_order = self.coef_index[-1]
        self.beta_index = []
        ind = 0
        for i in range(self.alpha_order):
            ind+=self.alpha_orders[i]
            self.beta_index += [ind+j for j in range(self.beta_order[i])]
            ind = ind + self.beta_order[i]

    def gradient(self):
        fitted = self.return_fitted()
        error = self.vts[self.alpha_order:,:] - fitted #shape=(T-p,N)
        
        grad_coefs = -2*np.sum(error[np.newaxis,:,:]*self.X,axis=(1,2))
        
        dwdgamma = self.dwdgamma() #shape = (num_stage,N,N)
        X_2 = []
        for t in range(self.alpha_order):
            X_2.append(self.vts[np.newaxis,self.alpha_order-t-1:-t-1,:]@dwdgamma[:self.beta_order[t]])
        X_2 = np.concatenate(X_2,axis=0)

        grad_gamma = 2*np.sum(error*np.sum(X_2*self.coefs[self.beta_index].reshape(-1,1,1),axis=0))
        return grad_coefs, grad_gamma, error

    def return_fitted(self):
        self.X = []
        for t in range(self.alpha_order):
            self.X.append(self.vts[np.newaxis,self.alpha_order-t-1:-t-1,:]@self.network.w_mats[1-self.alpha_orders[t]:1+self.beta_order[t]])
        self.X = np.concatenate(self.X,axis=0) #shape=(num_coef,T-p,N)
        if self.intercept:
            self.X = np.concatenate((self.X,self.X_intercept),axis=0)
        
        fitted = np.sum(self.X*self.coefs.reshape(-1,1,1),axis=0)
        
        return fitted

    def update_a(self,a_old):
        a = (1+np.sqrt(1+4*a_old**2))/2
        return a

    def fit(self,network,vts,grid_search=True,gamma_init=1,
            lr=1e-3,max_iter=10000,rtol=1e-10,new=True,accelerated=True,acc_param=1,stop_count_tol=10,
            search_start=0,search_end=10,search_num=1000,use_ls = True, l2_penal=0.,fit_mask=None):
        """
        Fit the model by either (i) grid search over gamma with least-squares coefficient
        updates, or (ii) joint gradient-based updates of coefficients and gamma.
    
        Parameters
        ----------
        network : object
            Network class object.
        vts : (T, N) ndarray
            Vector time series with T time points and N nodes. Column order must match
            the node order used by `network`.
        grid_search : bool, default=False
            If True, choose gamma by evaluating a grid on [search_start, search_end]
            (with `search_num` points), refitting coefficients for each gamma.
            If False, update gamma by gradient descent together with coefficients.
        gamma_init : float, default=1
            Initial value of the distance exponent gamma when `grid_search=False`.
        lr : float, default=1e-3
            Gradient step size used when `grid_search=False`.
        max_iter : int, default=10000
            Maximum number of gradient iterations when `grid_search=False`.
        rtol : float, default=1e-10
            Relative tolerance for early stopping when `grid_search=False`.
            The stopping check uses the relative change in consecutive entries of
            `self.losses`.
        new : bool, default=True
            If True, reinitialise internal state (data, caches, histories). If False,
            continue from existing state.
        accelerated : bool, default=True
            If True and `grid_search=False`, use Nesterov-style acceleration on the
            coefficient and gamma iterates via the sequence `a`.
        acc_param : float, default=1
            Initial acceleration parameter `a` used when `accelerated=True`.
        stop_count_tol : int, default=10
            Early-stopping patience for non-improving iterations when `grid_search=False`.
        search_start : float, default=0
            Left endpoint of the gamma grid when `grid_search=True`.
        search_end : float, default=10
            Right endpoint of the gamma grid when `grid_search=True`.
        search_num : int, default=1000
            Number of gamma values in the grid when `grid_search=True`.
        use_ls : bool, default=True
            Passed to `subfit()`. If True, use least-squares routines (`lstsq` / `lsqr`);
            otherwise use normal-equation solvers (`solve` / `spsolve`) depending on
            whether the design is dense or sparse.
        l2_penal : float, default=0.
            Passed to `subfit()`. If nonzero and the design is dense, perform ridge
            regression with penalty `l2_penal * I`. Used only when grid_search = True.
    
        Returns
        -------
        None
    
        Side Effects
        ------------
        Sets or updates the following attributes:
        - self.network : deep-copied from input `network` when `new=True`
        - self.vts, self.T, self.N
        - self.gamma, self.coefs
        - self.losses : list of values of np.sum(residual**2) appended during fitting
        - self.gamma_hist : list of gamma values visited during gradient iterations
          (only when `grid_search=False`)
        - self.network.w_mats[1:], self.w_norm, self.w_mats_unnorm
    
        Notes
        -----
        - When `grid_search=True`, gamma is chosen by minimising SSE over the grid, where
          SSE is computed from the vectorised response `y` and design `X` used in `subfit()`.
        - When `grid_search=False`, the method performs gradient updates using residuals
          computed from `return_fitted()`, and uses a best-so-far checkpointing rule with
          `stop_count_tol` patience.
        """
        self.gamma = gamma_init
    
        if fit_mask is not None:
            fit_mask = np.asarray(fit_mask,dtype=bool)
    
        if fit_mask is not None and not grid_search:
            raise ValueError("fit_mask currently supports grid_search=True only.")
    
        if new:
            self.rtol = rtol
            self.network = deepcopy(network)
            self.use_ls = use_ls
            self.l2_penal = l2_penal
            self.mask = self.network.d_mats!=0
            self.vts = vts
            self.X = self.transformVTS(self.vts)
            self.y = self.vts[self.alpha_order:,:].flatten("F")
            self.T,self.N = vts.shape
    
            if fit_mask is not None and len(fit_mask) != len(self.y):
                raise ValueError("fit_mask has the wrong length.")
            
            if self.intercept:
                if self.global_intercept:
                    self.X_intercept = np.ones((1,self.T-self.alpha_order,self.N))
                else:
                    self.X_intercept = np.zeros((self.N, self.T-self.alpha_order, self.N))
                    self.X_intercept[np.arange(self.N),:,np.arange(self.N)] = 1
                    
            self.update_gamma(self.gamma,fit_mask=fit_mask)
    
            if not grid_search:
                self.d_mats_log = np.zeros_like(self.network.d_mats)
                self.d_mats_log[self.mask] = np.log(self.network.d_mats[self.mask])
            
            self.losses = []
            self.best_loss = np.inf
            self.gamma_hist = [self.gamma]
            stop_count = 0
    
            if accelerated and not grid_search:
                self.a = acc_param
                coefs_old, gamma_old = self.coefs, self.gamma
                a_old = self.a
    
        if grid_search:
            self.gammas = np.linspace(search_start,search_end,search_num)
            self.losses = []
    
            for gamma in self.gammas:
                self.network.UpdateGamma(gamma)
                self.subfit(fit_mask=fit_mask)
    
                if fit_mask is None:
                    error = self.y-self.X@self.coefs
                else:
                    error = self.y[fit_mask]-self.X[fit_mask]@self.coefs
    
                self.losses.append(np.sum(error**2))
    
            self.gamma = self.gammas[np.argmin(self.losses)]
            self.network.UpdateGamma(self.gamma)
            self.subfit(fit_mask=fit_mask)
    
        else:
            for _ in range(max_iter):
                grad_coefs, grad_gamma, error = self.gradient()
                self.losses.append(np.sum(error**2))
    
                if _>1:
                    if stop_count > stop_count_tol or abs(self.losses[-1]-self.losses[-2])/abs(self.losses[-2])<=self.rtol:
                        self.coefs = best_coefs
                        self.gamma = best_gamma
                        break
    
                self.coefs -= (lr/(self.T-self.alpha_order)/self.N)*grad_coefs
                self.gamma -= (lr/(self.T-self.alpha_order)/self.N)*grad_gamma
                
                if accelerated:
                    self.a = self.update_a(self.a)
                    coefs_current, gamma_current = self.coefs, self.gamma
                    self.coefs = self.coefs + ((a_old-1)/self.a)*(self.coefs-coefs_old)
                    self.gamma = self.gamma + ((a_old-1)/self.a)*(self.gamma-gamma_old)
                    coefs_old, gamma_old = coefs_current, gamma_current
                    a_old = self.a
    
                if self.losses[-1] < self.best_loss:
                    stop_count = 0
                    self.best_loss = self.losses[-1]
                    best_coefs = self.coefs
                    best_gamma = self.gamma
                else:
                    stop_count += 1
                
                self.gamma_hist.append(self.gamma)
                self.network.w_mats[1:],self.w_norm, self.w_mats_unnorm = self.update_weights(self.gamma)
    
            if _ == max_iter-1:
                error = self.vts[self.alpha_order:,:]-self.return_fitted()
                self.losses.append(np.sum(error**2))
    
        self.network.UpdateGamma(float(self.gamma))
            
    def update_weights(self,gamma):
        w_mats_unnorm = np.zeros_like(self.network.d_mats)
        w_mats_unnorm[self.mask] = self.network.d_mats[self.mask]**(-gamma)
        w_norm = np.sum(w_mats_unnorm,axis=1,keepdims=True)
        w_mats = w_mats_unnorm/np.where(w_norm!=0,w_norm,1)
        return w_mats,w_norm,w_mats_unnorm

    def dwdgamma(self):
        grad = self.network.w_mats[1:]*(self.w_norm*self.d_mats_log-np.sum(self.w_mats_unnorm*self.d_mats_log,axis=1,keepdims=True))/np.where(self.w_norm!=0,self.w_norm,1)
        return grad

    def update_gamma(self,gamma,fit_mask=None):
        # compute weights and coefs, used for init and grid search
        self.network.w_mats[1:],self.w_norm, self.w_mats_unnorm = self.update_weights(gamma)
        self.subfit(fit_mask=fit_mask)

    def subfit(self,fit_mask=None):
        self.X = self.transformVTS(self.vts)
    
        if fit_mask is None:
            X = self.X
            y = self.y
        else:
            X = self.X[fit_mask]
            y = self.y[fit_mask]
    
        if self.global_alpha:
            if self.l2_penal != 0:
                self.coefs = np.linalg.solve(X.T@X + self.l2_penal*np.identity(X.shape[1]), X.T@y)
            else:
                if self.use_ls:
                    self.coefs = np.linalg.lstsq(X, y, rcond=None)[0]
                else:
                    self.coefs = np.linalg.solve(X.T@X, X.T@y)
        else:
            if self.use_ls:
                self.coefs = sparse.linalg.lsqr(X,y)[0]
            else:
                self.coefs = sparse.linalg.spsolve(X.T@X, X.T@y)

    def predict(self, length, nodes=None, vts_end =None):
        """
        Generate multi-step-ahead forecasts by iterating the fitted model.
    
        Parameters
        ----------
        length : int
            Number of steps ahead to forecast.
        nodes : array-like or None, default=None
            If None, return forecasts for all nodes. Otherwise, return forecasts for the
            selected node indices.
        vts_end : (alpha_order, N) ndarray or None, default=None
            Initial history used to start forecasting. If None, uses the last
            `alpha_order` observations from `self.vts`.
    
        Returns
        -------
        vts_pred : (length, N) ndarray or (length, len(nodes)) ndarray
            Forecasts for horizons 1..length. If `nodes` is provided, only those columns
            are returned.
        """
        if vts_end is None:
            vts_end = self.vts[-self.alpha_order:,:]
        vts_pred = np.zeros((length+self.alpha_order,self.network.size))
        vts_pred[:self.alpha_order,:] = vts_end
        for i in range(length):
            vts_pred[self.alpha_order+i,:] = self.transformVTS(vts_pred[i:self.alpha_order+i+1,:])@self.coefs
        if nodes is None:
            return vts_pred[self.alpha_order:,:]
        else:
            return vts_pred[self.alpha_order:,nodes]

    def validate(self, vts):
        """
        Compute the sum of squared prediction errors on a supplied time series.
    
        Parameters
        ----------
        vts : (T, N) ndarray
            Validation vector time series. Must have the same number of nodes N and
            column ordering as used in fitting.
    
        Returns
        -------
        sse : float
            Sum of squared errors, computed as np.sum((y - X @ self.coefs)**2) where
            X is the design matrix from `transformVTS(vts)` and
            y = vts[alpha_order:, :].flatten("F").
        """
        X = self.transformVTS(vts)
        y = vts[self.alpha_order:,:].flatten("F")
        return np.sum((y-X@self.coefs)**2)

    def compute_ci(self,level=0.95):
        p = self.alpha_order
        T,N = self.vts.shape
        n_time = T-p
    
        error = self.y-self.X@self.coefs
    
        if not hasattr(self,"d_mats_log"):
            self.d_mats_log = np.zeros_like(self.network.d_mats)
            self.d_mats_log[self.mask] = np.log(self.network.d_mats[self.mask])
    
        self.network.w_mats[1:],self.w_norm,self.w_mats_unnorm = self.update_weights(self.gamma)
    
        dwdgamma = self.dwdgamma()
    
        X_2 = []
        for t in range(self.alpha_order):
            X_2.append(self.vts[np.newaxis,p-t-1:-t-1,:]@dwdgamma[:self.beta_order[t]])
        X_2 = np.concatenate(X_2,axis=0)
    
        X_2 = np.sum(X_2*self.coefs[self.beta_index].reshape(-1,1,1),axis=0)
        X_2 = X_2.flatten("F")
    
        if sparse.issparse(self.X):
            X = sparse.hstack((self.X,sparse.csr_matrix(X_2.reshape(-1,1)))).toarray()
        else:
            X = np.column_stack((self.X,X_2))
    
        error = error.reshape((n_time,N),order="F")
    
        q = X.shape[1]
        U = np.zeros((q,q))
        R = np.zeros((q,q))
    
        for i in range(n_time):
            ind = i+n_time*np.arange(N)
            temp = X[ind,:]
    
            U += temp.T@temp
    
            temp = temp.T@error[i,:]
            R += np.outer(temp,temp)
    
        U = U/n_time
        R = R/n_time
    
        V = np.linalg.solve(U,R)
        V = np.linalg.solve(U,V.T).T
        V = 0.5*(V+V.T)
    
        self.se = np.sqrt(np.diag(V)/n_time)
    
        quantile = 1.959963984540054
        theta = np.concatenate((self.coefs,np.array([self.gamma])))
    
        self.CI = np.column_stack((
            theta-quantile*self.se,
            theta+quantile*self.se
        ))
    def loo(self,network,vts,grid_search=True,gamma_init=1,
            lr=1e-3,max_iter=10000,rtol=1e-10,accelerated=True,acc_param=1,stop_count_tol=10,
            search_start=0,search_end=10,search_num=1000,use_ls=True,l2_penal=0.):
        p = self.alpha_order
        T,N = vts.shape
        n_time = T-p
    
        y = vts[p:,:].flatten("F")
        errors = np.zeros((n_time,N))
    
        for i in range(n_time):
            fold_ind = i+n_time*np.arange(N)
    
            fit_mask = np.ones(len(y),dtype=bool)
            fit_mask[fold_ind] = False
    
            self.fit(network,vts,grid_search=grid_search,gamma_init=gamma_init,
                     lr=lr,max_iter=max_iter,rtol=rtol,new=True,accelerated=accelerated,acc_param=acc_param,stop_count_tol=stop_count_tol,
                     search_start=search_start,search_end=search_end,search_num=search_num,use_ls=use_ls,l2_penal=l2_penal,fit_mask=fit_mask)
    
            errors[i,:] = y[fold_ind]-self.X[fold_ind]@self.coefs
    
        return errors

    def gamma_cap(self,network,vts,gap,fit_mask=None,gap_rtol=1e-8,gap_atol=0.,gamma_tol=1e-6,doubling_init=1.,doubling_factor=2.,max_gamma=1e6):
        self.network = deepcopy(network)
        self.vts = np.asarray(vts,dtype=float)
        self.T,self.N = self.vts.shape
    
        p = self.alpha_order
        n_time = self.T-p
    
        self.y = self.vts[p:,:].flatten("F")
        y = self.y.copy()
    
        if fit_mask is not None:
            fit_mask = np.asarray(fit_mask,dtype=bool)
            y = y[fit_mask]
    
        stage_n = max(self.beta_order)
        d_mats = np.asarray(self.network.d_mats[:stage_n],dtype=float)
        distance_mask = d_mats>0
    
        w_inf = np.zeros_like(d_mats)
        gap_data = [[None for _ in range(self.N)] for _ in range(stage_n)]
    
        for r in range(stage_n):
            for n in range(self.N):
                ind = np.flatnonzero(distance_mask[r,:,n])
    
                if len(ind) == 0:
                    continue
    
                dists = d_mats[r,ind,n]
                min_local = dists==np.min(dists)
                gaps = np.log(dists/np.min(dists))
                k = np.sum(min_local)
    
                w_inf[r,ind[min_local],n] = 1./k
                gap_data[r][n] = (ind,gaps,min_local,k)
    
        w_mats_old = self.network.w_mats.copy()
        w_mats_temp = w_mats_old.copy()
        w_mats_temp[1:1+stage_n] = w_inf
        self.network.w_mats = w_mats_temp
        X_inf = self.transformVTS(self.vts)
        self.network.w_mats = w_mats_old
    
        if sparse.issparse(X_inf):
            X_inf = X_inf.toarray()
        else:
            X_inf = np.asarray(X_inf,dtype=float)
    
        if fit_mask is not None:
            X_inf = X_inf[fit_mask]
    
        singular_values = np.linalg.svd(X_inf,compute_uv=False)
        sigma_min = singular_values[-1]
        rank_tol = np.finfo(float).eps*max(X_inf.shape)*singular_values[0]
    
        if X_inf.shape[0]<X_inf.shape[1] or sigma_min<=rank_tol:
            raise ValueError("The design at gamma = infinity is rank deficient.")
    
        coef_inf = np.linalg.lstsq(X_inf,y,rcond=None)[0]
        residual_inf = y-X_inf@coef_inf
        loss_inf = np.sum(residual_inf**2)
    
        safety = gap_atol+gap_rtol*max(1.,abs(gap))
        usable_gap = gap-safety
    
        if usable_gap<=0:
            raise ValueError("The gap is not larger than the requested tolerance.")
    
        beta_blocks = []
        col = 0
    
        for t in range(p):
            col += self.alpha_orders[t] if self.global_alpha else self.alpha_orders[t]*self.N
    
            for r in range(self.beta_order[t]):
                if self.global_beta:
                    beta_blocks.append((col,t,r,True))
                    col += 1
                else:
                    beta_blocks.append((col,t,r,False))
                    col += self.N
    
        if self.intercept:
            col += 1 if self.global_intercept else self.N
    
        lagged_abs = [np.abs(self.vts[p-t-1:-t-1,:]) for t in range(p)]
        abs_X_inf = np.abs(X_inf)
        abs_coef_inf = np.abs(coef_inf)
        abs_residual_inf = np.abs(residual_inf)
    
        def improvement_bound(gamma):
            weight_envelopes = []
    
            for r in range(stage_n):
                envelope = np.zeros((self.N,self.N))
    
                for n in range(self.N):
                    if gap_data[r][n] is None:
                        continue
    
                    ind,gaps,min_local,k = gap_data[r][n]
                    nonmin_local = ~min_local
    
                    if not np.any(nonmin_local):
                        continue
    
                    q = np.exp(-gaps[nonmin_local]*gamma)
                    S = np.sum(q)
    
                    envelope[ind[min_local],n] = S/(k*(k+S))
                    envelope[ind[nonmin_local],n] = q/k
    
                weight_envelopes.append(envelope)
    
            M = np.zeros_like(X_inf)
    
            for start,t,r,is_global in beta_blocks:
                temp = lagged_abs[t]@weight_envelopes[r]
    
                if is_global:
                    block = temp.reshape((-1,1),order="F")
    
                    if fit_mask is not None:
                        block = block[fit_mask]
    
                    M[:,start] = block[:,0]
                else:
                    block = np.zeros((n_time*self.N,self.N))
    
                    for n in range(self.N):
                        ind = n*n_time+np.arange(n_time)
                        block[ind,n] = temp[:,n]
    
                    if fit_mask is not None:
                        block = block[fit_mask]
    
                    M[:,start:start+self.N] = block
    
            design_error = np.linalg.norm(M,2)
    
            if design_error>=sigma_min:
                return np.inf
    
            M_coef = M@abs_coef_inf
    
            return (
                2*abs_residual_inf@M_coef
                +(
                    np.linalg.norm(M.T@abs_residual_inf)
                    +np.linalg.norm(abs_X_inf.T@M_coef)
                    +np.linalg.norm(M.T@M_coef)
                )**2/(sigma_min-design_error)**2
            )
    
        bound = improvement_bound(0.)
    
        if bound<=usable_gap:
            gamma_cap = 0.
            certified = True
        else:
            gamma_l = 0.
            gamma_u = min(float(doubling_init),float(max_gamma))
            bound = improvement_bound(gamma_u)
    
            while bound>usable_gap and gamma_u<max_gamma:
                gamma_l = gamma_u
                gamma_u = min(doubling_factor*gamma_u,max_gamma)
                bound = improvement_bound(gamma_u)
    
            certified = bound<=usable_gap
    
            if certified:
                while gamma_u-gamma_l>gamma_tol:
                    gamma_middle = (gamma_l+gamma_u)/2.
                    bound_middle = improvement_bound(gamma_middle)
    
                    if bound_middle<=usable_gap:
                        gamma_u = gamma_middle
                    else:
                        gamma_l = gamma_middle
    
                gamma_cap = gamma_u
                bound = improvement_bound(gamma_cap)
            else:
                gamma_cap = float(max_gamma)
    
        self.loss_inf = float(loss_inf)
        self.gamma_cap_ = float(gamma_cap)
        self.gamma_cap_info = {
            "loss_inf":float(loss_inf),
            "gap":float(gap),
            "safety_margin":float(safety),
            "usable_gap":float(usable_gap),
            "sigma_min_limit":float(sigma_min),
            "improvement_bound_at_cap":float(bound),
            "certified":certified
        }
    
        return self.gamma_cap_

def gamma_pred_diff(beta_order,betas,w_mats_1,w_mats_2,autocovs):
    diff = 0
    ind_i=0
    for i in range(len(beta_order)):
        ind_j = 0
        for j in range(len(beta_order)):
            autocov = autocovs[np.abs(i-j)]
            for r in range(beta_order[i]):
                for s in range(beta_order[j]):
                    diff += betas[ind_i+r]*betas[ind_j+s]*np.trace((w_mats_1[r+1].T - w_mats_2[r+1].T)@autocov@np.transpose(w_mats_1[s+1].T - w_mats_2[s+1].T))
            ind_j += beta_order[j]
        ind_i += beta_order[i]
    return diff

def autocov_matrix(X, max_lag=None, demean=True):
    """
    Compute sample autocovariance matrices for a vector time series.

    Parameters
    ----------
    X : (T, N) ndarray
        Time series with T observations of N variables.
    max_lag : int or None
        Maximum lag to compute (default: T-1)
    demean : bool
        If True, subtract column means.

    Returns
    -------
    Gamma : list of np.ndarray
        List [Gamma(0), Gamma(1), ..., Gamma(max_lag)],
        each of shape (N, N).
    """
    X = np.asarray(X)
    T, N = X.shape
    if demean:
        X = X - X.mean(axis=0)
    if max_lag is None:
        max_lag = T - 1

    Gamma = []
    for h in range(max_lag + 1):
        # lagged cross-products
        G = (X[h:].T @ X[:T - h]) / (T - h)
        Gamma.append(G)
    return Gamma

from scipy.linalg import solve_discrete_lyapunov

def true_var_autocov(A_list, Sigma_eps, max_lag):
    """
    Compute true autocovariances Γ(h) for a stable VAR(p).

    Parameters
    ----------
    A_list : list of (N,N) arrays
        Coefficient matrices [A1,...,Ap].
    Sigma_eps : (N,N) array
        Innovation covariance.
    max_lag : int
        Maximum lag h to compute (≥ p).

    Returns
    -------
    Gamma : list of (N,N) arrays
        [Γ(0), Γ(1), ..., Γ(max_lag)]
    """
    N = A_list[0].shape[0]
    p = len(A_list)

    # Companion form
    Ac = np.block([
        [np.hstack(A_list)],
        [np.eye(N*(p-1)), np.zeros((N*(p-1), N))]
    ])
    B = np.zeros((N*p, N))
    B[:N, :] = np.eye(N)

    # Solve Lyapunov Σ_Y = A_c Σ_Y A_c' + B Σ_ε B'
    SigmaY = solve_discrete_lyapunov(Ac, B @ Sigma_eps @ B.T)

    # Extract Γ(0)
    Gamma0 = SigmaY[:N, :N]

    # Recursively compute Γ(h)
    Gamma = [Gamma0]
    for h in range(1, max_lag + 1):
        Ac_power = np.linalg.matrix_power(Ac, h)
        Gamma_h = Ac_power[:N, :N*p] @ SigmaY[:N*p, :N]
        Gamma.append(Gamma_h)

    return Gamma