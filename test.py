import sys
import os
import importlib

# ===== PARÂMETROS PERSONALIZÁVEIS =====
NUM_IMOVEIS = 50000       # 🔧 Quantos imóveis serão gerados
NUM_EPISODIOS = 500      # 🔁 Quantos passos de simulação
SAVE_PATH = "environments/GEO/tests/mapa_animado_v1.html"  # 📁 Caminho de saída do HTML
# =====================================
sys.path.append(os.path.abspath("."))
import environments.HomeChoice_v1 as home_env
importlib.reload(home_env)
HomeChoiceEnv = home_env.HomeChoiceEnv
# Instancia ambiente com número de imóveis definido
env = HomeChoiceEnv()
env.num_imoveis = NUM_IMOVEIS
env.market = env._generate_market()
obs = env.reset()
env.vendidos = []

# Guarda histórico para visualização
historico = []
for step in range(NUM_EPISODIOS):
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
    if action == 2 and hasattr(env, "last_sold"):
        env.vendidos.append(env.last_sold)
        delattr(env, "last_sold")
    # Salva histórico para visualização depois
    if step % 5 == 0:
        if env.current_step < len(env.market):
            prop = env.market[env.current_step]
            historico.append({
                "step": step,
                "prop": prop,
                "owned": prop in env.owned_properties,
                "sold": prop in env.vendidos
            })
    if done:
        break
# Renderiza uma única vez ao final
env.render_folium_timelapse_v1(historico, save_path=SAVE_PATH)
