import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
from smt.surrogate_models import KRG
from pydacefit.dace import DACE
from pydacefit.corr import corr_gauss
from pydacefit.regr import regr_constant
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from scipy.stats import norm
import torch
import tempfile
from utils.sampling import UniformPoint
from utils.ga_operators import GAreal
from utils.selection import kriging_selection
from models.ibnn import MultiTaskIBNN

class EMMOEA:
    def __init__(self, num_pop, num_obj, num_var, bounds, problem, surrogate='KRG', max_evals=400, gmax=10):
        self.N = num_pop
        self.M = num_obj
        self.D = num_var
        self.bounds = bounds
        self.problem = problem
        self.surrogate = surrogate
        self.num_evals = max_evals
        self.gmax = gmax
    
    def scale_and_evaluate(self, evaluate, Plhs):
        lower = self.bounds[0]
        upper = self.bounds[1]
        scaled_values = (upper - lower) * Plhs + lower
        Plhs_scaled = scaled_values
        return evaluate(scaled_values), Plhs_scaled
    
    def optimize(self):
        V = UniformPoint(self.N, self.M)
        NI = 100
        Plhs = UniformPoint(NI, self.D, 'Latin')
        TSObj, TSDec = self.scale_and_evaluate(self.problem, Plhs)
        lob = np.full(self.D, 1e-5)
        upb = np.full(self.D, 100)
        THETA = np.full((self.M, self.D), 5, dtype='float64')
        THETA0 = np.full(self.D, 5, dtype='float64')
        Kmodels = [None] * self.M
        if self.surrogate == 'GPytorch':
            lh = [None] * self.M
        PopDec = np.copy(TSDec)
        evals = self.N
        TS_objs = np.copy(TSObj)
        TS_decs = np.copy(TSDec)

        with tqdm(total=self.num_evals, initial=evals, desc="Number of evaluations") as pbar:
            while self.num_evals > evals:
                TSObj = np.copy(TS_objs)         
                TSDec = np.copy(TS_decs)
                if self.surrogate == 'MultiTaskIBNN':
                    # Создаём одну модель для ВСЕХ целевых функций
                    model_args = {"var_b": 1.0, "var_w": 1.0, "depth": 2}
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    # Нормализуем данные в [0, 1] и используем float64 для лучшей численной стабильности
                    x_min, x_max = TSDec.min(axis=0), TSDec.max(axis=0)
                    x_min = np.where(x_max == x_min, x_min - 1, x_min)  # Избежать деления на 0
                    x_normalized = (TSDec - x_min) / (x_max - x_min)
                    train_x = torch.tensor(x_normalized, dtype=torch.float64).to(device)
                    train_y = torch.tensor(TSObj, dtype=torch.float64).to(device)  # ВСЕ целевые функции!
                    Kmodels[0] = MultiTaskIBNN(model_args, self.D, self.M, device)
                    with tempfile.TemporaryDirectory() as tmpdir:
                        Kmodels[0].fit_and_save(train_x, train_y, tmpdir)
                    # Сохраняем параметры нормализации для предсказаний
                    self.x_min_ibnn = x_min
                    self.x_max_ibnn = x_max
                else:
                    for i in range(self.M):
                        if self.surrogate == 'DACE':
                            Kmodels[i] = DACE(regr=regr_constant, corr=corr_gauss, theta=THETA[i, :], thetaL=lob, thetaU=upb)
                            Kmodels[i].fit(TSDec, TSObj[:, i])
                            THETA[i, :] = Kmodels[i].model['theta']
                        elif self.surrogate == 'KRG':
                            Kmodels[i] = KRG(theta0=np.ones(self.D), print_global=False)
                            Kmodels[i].set_training_values(TSDec, TSObj[:, i])
                            Kmodels[i].train()
                            THETA[i, :] = Kmodels[i].optimal_theta
                g = 0
                while g < self.gmax:
                    OffDec = GAreal(PopDec, self.bounds)
                    PopDec = np.vstack((PopDec, OffDec))
                    N = PopDec.shape[0]
                    PopObj = np.zeros((N, self.M))
                    MSE = np.zeros((N, self.M))
                    
                    if self.surrogate == 'MultiTaskIBNN':
                        # Одна модель для всех целевых функций
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        # Нормализуем данные используя сохраненные параметры
                        PopDec_normalized = (PopDec - self.x_min_ibnn) / (self.x_max_ibnn - self.x_min_ibnn)
                        X_test = torch.tensor(PopDec_normalized, dtype=torch.float64).to(device)
                        posterior = Kmodels[0].posterior(X_test)
                        mean = posterior.mean.detach().cpu().numpy()
                        variance = posterior.variance.detach().cpu().numpy()
                        PopObj = mean
                        MSE = variance
                    else:
                        for i in range(N):
                            for j in range(self.M):
                                if self.surrogate == 'DACE':
                                    y_pred, mse_pred = Kmodels[j].predict(
                                        PopDec[i].reshape(1, self.D),
                                        return_mse=True
                                    )
                                    PopObj[i, j] = y_pred[0, 0]
                                    MSE[i, j]    = mse_pred[0, 0]
                                elif self.surrogate == 'KRG':
                                    PopObj[i][j] = Kmodels[j].predict_values(PopDec[i].reshape(-1, self.D)).item()
                                    MSE[i][j] = Kmodels[j].predict_variances(PopDec[i].reshape(-1, self.D)).item()
                                
                    index = kriging_selection(PopObj, V)
                    PopDec = PopDec[index]
                    g += 1
                min_objs = np.min(TSObj, axis=0)
                max_objs = np.max(TSObj, axis=0)
                TSObj = (TSObj - min_objs) / (max_objs - min_objs)
                Z = np.min(TSObj, axis=0)
                dc = np.linalg.norm(TSObj - Z, axis=1)
                ddt = cdist(TSObj, TSObj, metric='euclidean')
                np.fill_diagonal(ddt, np.inf)
                dd = np.min(ddt, axis=1)
                IP = dc - dd
                IPmin = np.min(IP)
                if self.surrogate == 'DACE':
                    IPmodel = DACE(regr=regr_constant, corr=corr_gauss, theta=THETA0, thetaL=lob, thetaU=upb)
                    IPmodel.fit(TSDec, IP)
                    THETA0 = IPmodel.model['theta']
                elif self.surrogate == 'KRG':
                    IPmodel = KRG(theta0=THETA0, print_global=False)
                    IPmodel.set_training_values(TSDec, IP)
                    IPmodel.train()
                    THETA0 = IPmodel.theta0
                elif self.surrogate == 'MultiTaskIBNN':
                    model_args = {"var_b": 1.0, "var_w": 1.0, "depth": 2}
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    # Нормализуем данные и используем float64
                    x_min_ip, x_max_ip = TSDec.min(axis=0), TSDec.max(axis=0)
                    x_min_ip = np.where(x_max_ip == x_min_ip, x_min_ip - 1, x_min_ip)
                    x_normalized_ip = (TSDec - x_min_ip) / (x_max_ip - x_min_ip)
                    train_x = torch.tensor(x_normalized_ip, dtype=torch.float64).to(device)
                    train_y = torch.tensor(IP.reshape(-1, 1), dtype=torch.float64).to(device)
                    IPmodel = MultiTaskIBNN(model_args, self.D, 1, device)
                    with tempfile.TemporaryDirectory() as tmpdir:
                        IPmodel.fit_and_save(train_x, train_y, tmpdir)
                    # Сохраняем параметры нормализации для IP
                    self.x_min_ip = x_min_ip
                    self.x_max_ip = x_max_ip
                sizePopDec = PopDec.shape[0]
                preIP = np.zeros((sizePopDec, 1))
                MSEIP = np.zeros((sizePopDec, 1))
                if self.surrogate == 'DACE':
                    preip, mseip = IPmodel.predict(PopDec, return_mse=True)
                    preIP = preip.flatten()
                    MSEIP = mseip.flatten()
                elif self.surrogate == 'KRG':
                    preip = IPmodel.predict_values(PopDec)
                    mseip = IPmodel.predict_variances(PopDec)
                    preIP = preip.flatten()
                    MSEIP = mseip.flatten()
                elif self.surrogate == 'MultiTaskIBNN':
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    # Нормализуем данные используя сохраненные параметры
                    PopDec_normalized_ip = (PopDec - self.x_min_ip) / (self.x_max_ip - self.x_min_ip)
                    X_test = torch.tensor(PopDec_normalized_ip, dtype=torch.float64).to(device)
                    posterior = IPmodel.posterior(X_test)
                    preIP = posterior.mean.detach().cpu().numpy().flatten()
                    MSEIP = posterior.variance.detach().cpu().numpy().flatten()
                s = np.sqrt(MSEIP)
                lamda = (IPmin - preIP) / s
                phi = norm.pdf(lamda)
                Phi = norm.cdf(lamda)
                EIP = (IPmin - preIP) * Phi + s * phi
                maxind = np.argmax(EIP)
                Popreal = PopDec[maxind]
                TSDec1 = np.vstack([TSDec, Popreal])
                _, indexes = np.unique(TSDec1, axis=0, return_index=True)
                if len(indexes) == TSDec1.shape[0]:
                    NewTSObj = self.problem(Popreal)
                    NewTSDec = np.copy(Popreal)
                    evals += 1
                    pbar.update(1)
                    TSObj = np.vstack([TSObj, NewTSObj])
                    TSDec = np.vstack([TSDec, NewTSDec])
                    TS_objs = np.vstack([TS_objs, NewTSObj])
                    TS_decs = np.vstack([TS_decs, NewTSDec])
                else:
                    print('Индивид не уникальный')
                nds = NonDominatedSorting()
                front_p = nds.do(TSObj)[0]
                TSndObj = TSObj[front_p]
                TSndDec = TSDec[front_p]
                min_TSndObj = np.min(TSndObj, axis=0)
                max_TSndObj = np.max(TSndObj, axis=0)
                TSndObj = (TSndObj - min_TSndObj) / (max_TSndObj - min_TSndObj)
                cos_dist = 1 - cdist(TSndObj, V, metric='cosine')
                AngleND = np.arccos(cos_dist)
                ValueAngle = np.min(AngleND, axis=1)
                associate = np.argmin(AngleND, axis=1)
                TSsec = np.empty((0, self.D))
                for i in np.unique(associate):
                    current = np.where(associate == i)[0]
                    minindc = np.argmin(ValueAngle[current])
                    TSsect = TSndDec[current[minindc], :]
                    TSsec = np.vstack([TSsec, TSsect])
                TSsec = np.array(TSsec)
                PopDec = np.vstack([PopDec, TSsec])
                PopDec = np.unique(PopDec, axis=0)
        
        return TS_decs, TS_objs
        
