import time
import random
import numpy as np
from mip import Model, xsum, MINIMIZE, OptimizationStatus, CONTINUOUS, Column

# =============================================================================
# 1. GERADOR DE INSTÂNCIAS
# =============================================================================
class CInstance:
    def __init__(self, ni=10, nt=10, seed=2025):
        
        gamma_map = {5: 200, 10: 300, 15: 500, 20: 1000}
        
        if ni not in gamma_map:
             gamma_map[ni] = 200 * (ni/5)

        np.random.seed(seed)
        random.seed(seed)
        
        self.ni, self.nt = ni, nt
        self.I, self.T = range(ni), range(nt)
        
        # Geração dos parametros aleatórios
        self.d = np.random.randint(10, 46, (ni, nt))     # Demanda
        self.p = np.random.randint(20, 151, (ni, nt))    # Custo Produção
        self.f = np.random.randint(2000, 5000, (ni, nt)) # Setup
        self.h = np.floor(self.p * 0.10)                 # Estocagem (10% de p)
        
        self.gamma = np.full(nt, gamma_map.get(ni, 200)) # Vetor de Capacidade
        
        # Big M para o modelo compacto
        self.BigM = np.sum(self.d) * 1.5

# =============================================================================
# 2. MODELO COMPACTO (RELAXAÇÃO LINEAR)
# =============================================================================
def solve_compact_lp(inst):
    m = Model('CLSP_Compact', sense=MINIMIZE, solver_name='CBC')
    m.verbose = 0
    
    # Variáveis
    x = [[m.add_var(name=f'x_{i}_{t}', lb=0) for t in inst.T] for i in inst.I]
    s = [[m.add_var(name=f's_{i}_{t}', lb=0) for t in inst.T] for i in inst.I]
    # Relaxação Linear: y contínuo [0, 1]
    y = [[m.add_var(name=f'y_{i}_{t}', lb=0, ub=1) for t in inst.T] for i in inst.I]

    # Função Objetivo (1)
    m.objective = xsum(inst.h[i][t]*s[i][t] + inst.p[i][t]*x[i][t] + inst.f[i][t]*y[i][t] 
                       for i in inst.I for t in inst.T)

    # Restrições
    for i in inst.I:
        for t in inst.T:
            s_prev = s[i][t-1] if t > 0 else 0
            # (2) Balanço
            m += s_prev + x[i][t] == inst.d[i][t] + s[i][t]
            # (3) Setup (Relaxado)
            m += x[i][t] <= inst.BigM * y[i][t]

    # (4) Capacidade Global
    for t in inst.T:
        m += xsum(s[i][t] for i in inst.I) <= inst.gamma[t]

    m.optimize()
    return m.objective_value

# =============================================================================
# 3. PRICING VIA PROGRAMAÇÃO DINÂMICA
# =============================================================================
def solve_pricing_dp(inst, duals_u):
    """
    Retorna:
    - min_path_val: Custo total dos caminhos mínimos (já descontando u_t * s_it)
    - global_storage_plan: Vetor de estoque agregado para a nova coluna (soma de s_it)
    - total_real_cost: Custo Primal da nova coluna (sem duais)
    """
    total_min_path_cost = 0
    total_real_cost = 0
    global_storage_plan = np.zeros(inst.nt) 

    for i in inst.I:
        
        h_mod = [inst.h[i][t] - duals_u[t] for t in inst.T]
        
        min_cost = [float('inf')] * (inst.nt + 1)
        min_cost[0] = 0.0
        parent = [-1] * (inst.nt + 1)
        arc_data = {} 

        
        for t in inst.T: 
            if min_cost[t] == float('inf'): continue
            
            cum_demand = 0
            cum_holding_mod = 0
            cum_holding_real = 0
            
            
            for tau in range(t, inst.nt):
                d_curr = inst.d[i][tau]
                cum_demand += d_curr
                
               
                for k in range(t, tau):
                    cum_holding_mod += h_mod[k] * d_curr
                    cum_holding_real += inst.h[i][k] * d_curr
                
                prod_setup_cost = inst.f[i][t] + (inst.p[i][t] * cum_demand)
                
                cost_mod = prod_setup_cost + cum_holding_mod
                cost_real = prod_setup_cost + cum_holding_real
                
                next_node = tau + 1
                
                if min_cost[t] + cost_mod < min_cost[next_node]:
                    min_cost[next_node] = min_cost[t] + cost_mod
                    parent[next_node] = t
                    
                    
                    s_prof = np.zeros(inst.nt)
                    curr_stock = 0
                    for k in range(tau, t, -1):
                        curr_stock += inst.d[i][k]
                        s_prof[k-1] = curr_stock
                    
                    arc_data[(t, next_node)] = (cost_real, s_prof)

        
        total_min_path_cost += min_cost[inst.nt]
        curr = inst.nt
        while curr > 0:
            prev = parent[curr]
            c_real, s_prof = arc_data[(prev, curr)]
            total_real_cost += c_real
            global_storage_plan += s_prof
            curr = prev

    return total_min_path_cost, global_storage_plan, total_real_cost

# =============================================================================
# 4. ALGORITMO DANTZIG-WOLFE (Problema Mestre)
# =============================================================================
def solve_dantzig_wolfe(inst):
    mp = Model('DW_Master', sense=MINIMIZE, solver_name='CBC')
    mp.verbose = 0
    
    
    dummy = mp.add_var(obj=1e9, lb=0, name="lambda_dummy")
    
    
    constrs_cap = []
    for t in inst.T:
        c = mp.add_constr(0 * dummy <= inst.gamma[t], name=f"cap_{t}")
        constrs_cap.append(c)
        
    
    constr_conv = mp.add_constr(dummy == 1, name="conv")
    
    iter_count = 0
    
    while True:
        iter_count += 1
        mp.optimize()
        
        if mp.status != OptimizationStatus.OPTIMAL:
            return -1
            
        
        u_t = [c.pi for c in constrs_cap]
        u_0 = constr_conv.pi
        
        
        min_path_val, col_s, col_cost = solve_pricing_dp(inst, u_t)
        
        
        reduced_cost = min_path_val - u_0
        
        if reduced_cost >= -1e-4:
            
            return mp.objective_value
            
        
        
        
        constrs_list = [constr_conv]  
        coeffs_list = [1.0]           
        
        
        for t in inst.T:
            if col_s[t] > 1e-6:
                constrs_list.append(constrs_cap[t])
                coeffs_list.append(col_s[t])
        
        
        col = Column(constrs=constrs_list, coeffs=coeffs_list)
        mp.add_var(obj=col_cost, lb=0, column=col, name=f"lambda_{iter_count}")

# =============================================================================
# 5. EXECUÇÃO DOS EXPERIMENTOS
# =============================================================================
def main():
    print(f"{'='*80}")
    print(f"{'COMPARATIVO: RELAXAÇÃO LINEAR vs DANTZIG-WOLFE (10 Instâncias)':^80}")
    print(f"{'='*80}")
    print(f"{'#':<3} | {'Size':<8} | {'LP Bound':<12} | {'DW Bound':<12} | {'Gap (%)':<8} | {'Tempo':<8}")
    print("-" * 80)
    
    ni, nt = 10, 10
    
    for k in range(10):
        
        seed_k = 2025 + k 
        inst = CInstance(ni, nt, seed=seed_k)
        
        t0 = time.time()
        
        # 1. Relaxação Linear
        lp_val = solve_compact_lp(inst)
        
        # 2. Dantzig-Wolfe
        dw_val = solve_dantzig_wolfe(inst)
        
        tf = time.time()
        
        gap = 0.0
        if dw_val > 0 and lp_val > 0:
            
            gap = (dw_val - lp_val) / lp_val * 100
        elif dw_val == -1:
            dw_val = float('inf')
            
        print(f"{k+1:<3} | {ni}x{nt:<5} | {lp_val:<12.2f} | {dw_val:<12.2f} | {gap:<8.2f} | {tf-t0:<8.2f}")
        
    print("-" * 80)

if __name__ == "__main__":
    main()
