import sys
import os
import importlib

# ===== PARÂMETROS PERSONALIZÁVEIS =====
NUM_IMOVEIS   = 10000
NUM_EPISODIOS = 100
NUM_STEPS     = 2000
# =====================================

sys.path.append(os.path.abspath("."))

import environments.HomeChoice_v2 as home_env
importlib.reload(home_env)
HomeChoiceEnv = home_env.HomeChoiceEnv

# Instancia ambiente com número de imóveis e passos definidos
env = HomeChoiceEnv(max_steps=NUM_STEPS, render_mode='human')
env.num_imoveis = NUM_IMOVEIS
env.market = env._generate_market()
obs, _ = env.reset()

# Executa uma simulação para preencher o histórico
for _ in range(NUM_EPISODIOS):
    action = env.action_space.sample()
    obs, reward, done, _, _ = env.step(action)
    if done:
        break

# Renderiza mapa + gráficos de saldo e patrimônio
env.render_map_v0()
