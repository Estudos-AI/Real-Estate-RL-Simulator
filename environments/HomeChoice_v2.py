import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from shapely.geometry import Polygon, Point
import random
from typing import Optional
import geopandas as gpd
from pathlib import Path
from gymnasium import Env, spaces
from environments.GEO.maps.SP import distritos

class HomeChoiceEnv(Env):
    def __init__(self, render_mode='human', num_imoveis=10000, max_steps=20, target_wealth=1000000, verbose=True, price_update_freq=6):
        super().__init__()
        self.render_mode = render_mode
        self.num_imoveis = num_imoveis
        self.max_steps = max_steps
        self.target_wealth = target_wealth
        self.initial_cash = 100000
        self.bairro_poligonos = self._mapear_bairros_para_poligonos(distritos)
        self.idh_bairros = self._load_idh_bairros()
        self.market = self._generate_market()
        self.verbose = verbose
        self.price_update_freq = price_update_freq
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self._initialize_episode()

    def _initialize_episode(self):
        self.current_step = 0
        self.cash = self.initial_cash
        self.owned_properties = []
        self.history = []
        self.wait_steps = 0
        self.last_wealth = self._get_total_wealth()
        for prop in self.market:
            prop['status'] = 'disponível'

    def _mapear_bairros_para_poligonos(self, distritos):
        return {d["nome"].strip().upper(): d["poligono"] for d in distritos}

    def _load_idh_bairros(self):
        return {"AGUA RASA": 0.869, "ALTO DE PINHEIROS": 0.942, "ANHANGUERA": 0.731, "ARICANDUVA": 0.758, "ARTUR ALVIM": 0.804,
                "BARRA FUNDA": 0.889, "BELA VISTA": 0.889, "BELEM": 0.869, "BOM RETIRO": 0.889, "BRAS": 0.869,
                "BRASILANDIA": 0.762, "BUTANTA": 0.859, "CACHOEIRINHA": 0.799, "CAMBUCI": 0.889, "CAMPO BELO": 0.909,
                "CAMPO GRANDE": 0.909, "CAMPO LIMPO": 0.783, "CANGAIBA": 0.804, "CAPAO REDONDO": 0.783, "CARRAO": 0.758,
                "CASA VERDE": 0.799, "CIDADE ADEMAR": 0.758, "CIDADE DUTRA": 0.758, "CIDADE LIDER": 0.758, "CIDADE TIRADENTES": 0.708,
                "CONSOLACAO": 0.889, "CURSINO": 0.824, "ERMELINO MATARAZZO": 0.777, "FREGUESIA DO O": 0.762, "GRAJAU": 0.758,
                "GUAIANASES": 0.713, "IGUATEMI": 0.732, "IPIRANGA": 0.824, "ITAIM BIBI": 0.942, "ITAIM PAULISTA": 0.725,
                "ITAQUERA": 0.758, "JABAQUARA": 0.816, "JACANA": 0.869, "JAGUARA": 0.787, "JAGUARE": 0.787, "JARAGUA": 0.787,
                "JARDIM ANGELA": 0.716, "JARDIM HELENA": 0.736, "JARDIM PAULISTA": 0.942, "JARDIM SAO LUIS": 0.716,
                "JOSE BONIFACIO": 0.758, "LAJEADO": 0.713, "LAPA": 0.906, "LIBERDADE": 0.889, "LIMAO": 0.799,
                "MANDAQUI": 0.869, "MARSILAC": 0.708, "MOEMA": 0.938, "MOOCA": 0.869, "MORUMBI": 0.859, "PARELHEIROS": 0.708,
                "PARI": 0.869, "PARQUE DO CARMO": 0.758, "PEDREIRA": 0.758, "PENHA": 0.804, "PERDIZES": 0.906,
                "PERUS": 0.731, "PINHEIROS": 0.942, "PIRITUBA": 0.787, "PONTE RASA": 0.777, "RAPOSO TAVARES": 0.859,
                "REPUBLICA": 0.889, "RIO PEQUENO": 0.859, "SACOMA": 0.824, "SANTA CECILIA": 0.889, "SANTANA": 0.869,
                "SANTO AMARO": 0.909, "SAO DOMINGOS": 0.787, "SAO LUCAS": 0.758, "SAO MATEUS": 0.732, "SAO MIGUEL": 0.736,
                "SAO RAFAEL": 0.732, "SAPOPEMBA": 0.758, "SAUDE": 0.938, "SE": 0.889, "SOCORRO": 0.758, "TATUAPE": 0.869,
                "TREMEMBE": 0.869, "TUCURUVI": 0.869, "VILA ANDRADE": 0.783, "VILA CURUCA": 0.725, "VILA FORMOSA": 0.758,
                "VILA GUILHERME": 0.869, "VILA JACUI": 0.736, "VILA LEOPOLDINA": 0.906, "VILA MARIA": 0.869,
                "VILA MARIANA": 0.938, "VILA MATILDE": 0.804, "VILA MEDEIROS": 0.869, "VILA PRUDENTE": 0.758,
                "VILA SONIA": 0.859}

    def ponto_aleatorio_em_poligono(self, poligono, tentativas=100):
        poly = Polygon(poligono)
        minx, miny, maxx, maxy = poly.bounds
        for _ in range(tentativas):
            x, y = random.uniform(minx, maxx), random.uniform(miny, maxy)
            if poly.contains(Point(x, y)):
                return (x, y)
        return ((minx + maxx) / 2, (miny + maxy) / 2)

    def _generate_market(self):
        market = []
        bairros = list(self.idh_bairros.keys())
        coords_usadas = set()
        for i in range(self.num_imoveis):
            while True:
                bairro = np.random.choice(bairros)
                idh = self.idh_bairros.get(bairro, 0.8)
                poligono = self.bairro_poligonos.get(bairro.upper())
                if not poligono:
                    continue
                ponto = self.ponto_aleatorio_em_poligono(poligono)
                rounded = (round(ponto[0], 6), round(ponto[1], 6))
                if rounded not in coords_usadas:
                    coords_usadas.add(rounded)
                    break

            preco = int(np.interp(idh, [0.7, 0.95], [20000, 200000]))
            market.append({
                "id": i,
                "bairro": bairro,
                "idh_microrregiao": idh,
                "pos": rounded,
                "status": "disponível",
                "preco": preco
            })

        return market

    def _get_observation(self):
        if self.current_step >= len(self.market):
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        prop = self.market[self.current_step]
        x, y = prop['pos']
        idh = prop['idh_microrregiao']
        return np.array([x / -46.3, y / -23.3, idh], dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self._initialize_episode()
        return self._get_observation(), {}

    def _get_total_wealth(self):
        total_value = self.cash + sum(p['preco'] for p in self.owned_properties)
        return total_value

    def _update_prices(self):
        for prop in self.market:
            delta = np.random.uniform(0.95, 1.05)
            prop['preco'] = int(prop['preco'] * delta)

    def step(self, action):
        prop = self.market[self.current_step]
        preco = prop['preco']
        prev_wealth = self._get_total_wealth()

        if action == 0 and self.cash >= preco:
            prop['status'] = 'comprado'
            self.owned_properties.append(prop)
            self.cash -= preco
            if self.verbose:
                print(f"[STEP {self.current_step}] Comprou imóvel {prop['id']} por R${preco:,.2f}")

        elif action == 2 and self.owned_properties:
            sold = self.owned_properties.pop(0)
            sold['status'] = 'vendido'
            self.cash += sold['preco']
            if self.verbose:
                print(f"[STEP {self.current_step}] Vendeu imóvel {sold['id']} por R${sold['preco']:,.2f}")

        elif action == 1 and self.verbose:
            print(f"[STEP {self.current_step}] Aguardou (sem ação)")

        self.current_step += 1
        if self.current_step % self.price_update_freq == 0:
            self._update_prices()

        new_wealth = self._get_total_wealth()
        gain = new_wealth - prev_wealth

        reward = -0.5
        if gain > 0:
            reward += np.log1p(gain)

        self.history.append({
            'step': self.current_step,
            'cash': self.cash,
            'wealth': new_wealth
        })

        done = new_wealth >= self.target_wealth or self.current_step >= self.max_steps

        if self.verbose:
            print(f"[STEP {self.current_step}] Saldo: R${self.cash:,.2f} | Patrimônio: R${new_wealth:,.2f} | Recompensa: {reward:.4f}")
            print(f"Comprados: {len([p for p in self.market if p['status'] == 'comprado'])} | Vendidos: {len([p for p in self.market if p['status'] == 'vendido'])}")

        return self._get_observation(), reward, done, False, {}

    def render_map_v0(self):
        if self.render_mode != 'human':
            return

        if not self.history:
            print("⚠️ Nenhum dado de histórico disponível. Execute pelo menos um passo com env.step() antes de renderizar.")
            return

        gdf = gpd.read_file("environments/GEO/raw/distritos.geojson")
        df = pd.DataFrame(self.market)
        df[['x', 'y']] = pd.DataFrame(df['pos'].tolist(), index=df.index)
        cores = {'disponível': 'lightgrey', 'comprado': 'green', 'vendido': 'red'}

        fig, axs = plt.subplots(1, 3, figsize=(18, 8))

        # Mapa
        gdf.boundary.plot(ax=axs[0], linewidth=0.8, color='black')
        axs[0].scatter(df['x'], df['y'], s=20, c=[cores[status] for status in df['status']])
        wealth = self._get_total_wealth()
        axs[0].set_title(f"Último passo | Saldo: R${self.cash:,.2f} | Patrimônio: R${wealth:,.2f}", fontsize=12)

        # Saldo ao longo do tempo
        hist_df = pd.DataFrame(self.history)
        axs[1].plot(hist_df['step'], hist_df['cash'], label='Saldo', color='blue')
        axs[1].set_title("Saldo ao longo do tempo")
        axs[1].set_xlabel("Passo")
        axs[1].set_ylabel("R$")
        axs[1].grid(True)

        # Patrimônio ao longo do tempo
        axs[2].plot(hist_df['step'], hist_df['wealth'], label='Patrimônio', color='green')
        axs[2].set_title("Patrimônio ao longo do tempo")
        axs[2].set_xlabel("Passo")
        axs[2].set_ylabel("R$")
        axs[2].grid(True)

        plt.tight_layout()
        plt.show()
    
    def render_timelapse_ep(self, episodios_hist):

        gdf = gpd.read_file("environments/GEO/raw/distritos.geojson")
        df_market = pd.DataFrame(self.market)
        df_market[['x', 'y']] = pd.DataFrame(df_market['pos'].tolist(), index=df_market.index)
        cores = {'disponível': 'blue', 'comprado': 'green', 'vendido': 'red'}

        fig, axs = plt.subplots(1, 3, figsize=(18, 8))

        def update(frame):
            axs[0].cla()
            axs[1].cla()
            axs[2].cla()

            # Mapa
            gdf.boundary.plot(ax=axs[0], linewidth=0.8, color='black')
            axs[0].scatter(df_market['x'], df_market['y'], s=20, c=[cores[status] for status in df_market['status']])
            axs[0].set_title(f"Episódio {frame+1}")

            # Histórico do episódio atual
            hist_df = pd.DataFrame(episodios_hist[frame])
            axs[1].plot(hist_df['step'], hist_df['cash'], color='blue')
            axs[1].set_title("Saldo")
            axs[1].grid(True)

            axs[2].plot(hist_df['step'], hist_df['wealth'], color='green')
            axs[2].set_title("Patrimônio")
            axs[2].grid(True)

            return axs

        ani = FuncAnimation(fig, update, frames=len(episodios_hist), interval=800, repeat=False)
        plt.tight_layout()
        plt.show()
