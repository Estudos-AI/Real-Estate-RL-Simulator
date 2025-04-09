

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pandas as pd
import matplotlib.pyplot as plt
import time
import pygame
from shapely.geometry import Polygon, Point
import random
import os
import importlib
import sys
import geopandas as gpd
from environments.GEO.maps.SP import distritos
import folium
from folium.plugins import MarkerCluster, TimestampedGeoJson    
import json
import datetime
from pathlib import Path

###################################################################################################################
class HomeChoiceEnv(gym.Env):
    """
    Simulador de investimento imobiliário na cidade de São Paulo.
    O agente deve comprar e vender imóveis para atingir R$ 1.000.000.
    O mercado é dinâmico, com valorização e desvalorização dos imóveis baseada em características reais.
    """
    def __init__(self, render_mode='human'):
        super().__init__()
        self.bairro_poligonos = self._mapear_bairros_para_poligonos(distritos)
        self.render_mode        = render_mode
        self.history            = []  # Histórico para renderização gráfica
        self.fig, self.ax       = None, None
        self.initial_cash       = 100000 # Saldo inicial do agente
        self.cash               = 100000 
        self.owned_properties   = []  # Lista de imóveis comprados
        self.current_step       = 0
        self.waiting_steps      = 0 
        self.num_imoveis        = 10000 # Número de imóveis no mercado
        self.action_space       = spaces.Discrete(3) # Espaço de Ação: 0 = Comprar, 1 = Esperar, 2 = Vender
        self.observation_space  = spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32) # Espaço de Observação: [preço do imóvel, metragem, IDH, taxa de criminalidade, infraestrutura, saldo do agente]
        self.idh_bairros        = {
            "AGUA RASA": 0.869,"ALTO DE PINHEIROS": 0.942,"ANHANGUERA": 0.731,
            "ARICANDUVA": 0.758,"ARTUR ALVIM": 0.804,"BARRA FUNDA": 0.889,
            "BELA VISTA": 0.889,"BELEM": 0.869,"BOM RETIRO": 0.889,
            "BRAS": 0.869,"BRASILANDIA": 0.762,"BUTANTA": 0.859,
            "CACHOEIRINHA": 0.799,"CAMBUCI": 0.889,"CAMPO BELO": 0.909,
            "CAMPO GRANDE": 0.909,"CAMPO LIMPO": 0.783,"CANGAIBA": 0.804,
            "CAPAO REDONDO": 0.783,"CARRAO": 0.758,"CASA VERDE": 0.799,
            "CIDADE ADEMAR": 0.758,"CIDADE DUTRA": 0.758,"CIDADE LIDER": 0.758,
            "CIDADE TIRADENTES": 0.708,"CONSOLACAO": 0.889,"CURSINO": 0.824,
            "ERMELINO MATARAZZO": 0.777,"FREGUESIA DO O": 0.762,"GRAJAU": 0.758,
            "GUAIANASES": 0.713,"IGUATEMI": 0.732,"IPIRANGA": 0.824,
            "ITAIM BIBI": 0.942,"ITAIM PAULISTA": 0.725,"ITAQUERA": 0.758,
            "JABAQUARA": 0.816,"JACANA": 0.869,"JAGUARA": 0.787,
            "JAGUARE": 0.787,"JARAGUA": 0.787,"JARDIM ANGELA": 0.716,
            "JARDIM HELENA": 0.736,"JARDIM PAULISTA": 0.942,"JARDIM SAO LUIS": 0.716,
            "JOSE BONIFACIO": 0.758,"LAJEADO": 0.713,"LAPA": 0.906,
            "LIBERDADE": 0.889,"LIMAO": 0.799,"MANDAQUI": 0.869,
            "MARSILAC": 0.708,"MOEMA": 0.938,"MOOCA": 0.869,
            "MORUMBI": 0.859,"PARELHEIROS": 0.708,"PARI": 0.869,
            "PARQUE DO CARMO": 0.758,"PEDREIRA": 0.758,"PENHA": 0.804,
            "PERDIZES": 0.906,"PERUS": 0.731,"PINHEIROS": 0.942,
            "PIRITUBA": 0.787,"PONTE RASA": 0.777,"RAPOSO TAVARES": 0.859,
            "REPUBLICA": 0.889,"RIO PEQUENO": 0.859,"SACOMA": 0.824,
            "SANTA CECILIA": 0.889,"SANTANA": 0.869,"SANTO AMARO": 0.909,
            "SAO DOMINGOS": 0.787,"SAO LUCAS": 0.758,"SAO MATEUS": 0.732,
            "SAO MIGUEL": 0.736,"SAO RAFAEL": 0.732,"SAPOPEMBA": 0.758,
            "SAUDE": 0.938,"SE": 0.889,"SOCORRO": 0.758,"TATUAPE": 0.869,
            "TREMEMBE": 0.869,"TUCURUVI": 0.869,"VILA ANDRADE": 0.783,
            "VILA CURUCA": 0.725,"VILA FORMOSA": 0.758,"VILA GUILHERME": 0.869,
            "VILA JACUI": 0.736,"VILA LEOPOLDINA": 0.906,"VILA MARIA": 0.869,
            "VILA MARIANA": 0.938,"VILA MATILDE": 0.804,"VILA MEDEIROS": 0.869,
            "VILA PRUDENTE": 0.758,"VILA SONIA": 0.859
            }
        self._used_coords = set()  # Conjunto para coordenadas já usadas
        self.market             = self._generate_market()  # Gera o mercado inicial

###################################################################################################################

    def _generate_market(self):
        path_json = Path("environments/GEO/tests/imoveis_fixos.json")

        if path_json.exists():
            print("📥 Carregando imóveis fixos do arquivo...")
            with open(path_json, "r", encoding="utf-8") as f:
                return json.load(f)


        print("🛠️ Gerando novos imóveis...")
        market = []
        bairros = list(self.idh_bairros.keys())
        coords_usadas = set()

        for i in range(self.num_imoveis): 
            if i % 1000 == 0:
                print(f"🛠️  Gerando imóvel {i} / {self.num_imoveis}")

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

            # Escolhe tipo de imóvel conforme IDH
            if idh > 0.85:
                tipo_imovel = np.random.choice(["Apartamento Padrão", "Casa de Luxo", "Cobertura"], p=[0.5, 0.3, 0.2])
            elif idh > 0.75:
                tipo_imovel = np.random.choice(["Casa Popular", "Apartamento Padrão", "Casa de Luxo"], p=[0.3, 0.5, 0.2])
            else:
                tipo_imovel = np.random.choice(["Casa Popular", "Apartamento Padrão"], p=[0.7, 0.3])

            preco_m2_base = np.interp(idh, [0.7, 0.95], [2000, 15000])

            # Define metragem e preço
            if tipo_imovel == "Casa Popular":
                metragem = np.random.randint(80, 151)
                preco = int(metragem * preco_m2_base * np.random.uniform(0.9, 1.1))
                condominio = 0
            elif tipo_imovel == "Apartamento Padrão":
                metragem = np.random.randint(50, 101)
                preco = int(metragem * preco_m2_base * np.random.uniform(0.9, 1.2))
                condominio = np.random.randint(500, 1501)
            elif tipo_imovel == "Casa de Luxo":
                metragem = np.random.randint(200, 501)
                preco = int(metragem * preco_m2_base * np.random.uniform(1.0, 1.3))
                condominio = 0
            else:
                metragem = np.random.randint(150, 401)
                preco = int(metragem * preco_m2_base * np.random.uniform(1.2, 1.5))
                condominio = np.random.randint(2000, 5001)

            infraestrutura = np.interp(idh, [0.7, 0.95], [0.3, 1.0])
            taxa_criminalidade = np.interp(idh, [0.7, 0.95], [1.0, 0.2])
            demanda = int(np.interp(idh, [0.7, 0.95], [300, 1000]) * np.random.uniform(0.8, 1.2))

            property_data = {
                "id": i,
                "tipo": tipo_imovel,
                "bairro": bairro,
                "idh_microrregiao": idh,
                "metragem": metragem,
                "preco": preco,
                "condominio": condominio,
                "taxa_criminalidade": taxa_criminalidade,
                "infraestrutura": infraestrutura,
                "demanda": demanda,
                "tempo_no_mercado": 0,
                "pos": rounded
            }

            market.append(property_data)

        # Salva para reuso
        path_json.parent.mkdir(parents=True, exist_ok=True)
        with open(path_json, "w", encoding="utf-8") as f:
            json.dump(market, f, indent=2)

        print(f"✅ {len(market)} imóveis salvos em '{path_json}'")
        return market

###################################################################################################################

    def _apply_market_events(self):
        """Aplica eventos aleatórios que afetam o mercado imobiliário."""
        event = np.random.choice(["crise", "metrô", "shopping", "criminalidade", "neutro"], p=[0.15, 0.2, 0.2, 0.15, 0.3])

        for prop in self.market:
            if event == "crise":
                prop["preco"] *= np.random.uniform(0.85, 0.95)  # Queda de preços
            elif event == "metrô" and prop["infraestrutura"] > 0.8:
                prop["preco"] *= np.random.uniform(1.1, 1.3)  # Valorização nas áreas bem servidas
            elif event == "shopping" and prop["demanda"] > 500:
                prop["preco"] *= np.random.uniform(1.05, 1.2)  # Aumento da demanda
            elif event == "criminalidade" and prop["taxa_criminalidade"] > 0.7:
                prop["preco"] *= np.random.uniform(0.7, 0.9)  # Desvalorização em bairros perigosos
###################################################################################################################

    def _get_observation(self):
        """Retorna o estado atual do ambiente como um vetor normalizado."""
        if self.current_step >= len(self.market):
            return np.zeros(self.observation_space.shape)

        prop = self.market[self.current_step]
        price = prop["preco"] / 5000000  # Normaliza para [0, 1]
        demand = prop["demanda"] / 1000
        idh = prop["idh_microrregiao"]
        crime = prop["taxa_criminalidade"]
        infra = prop["infraestrutura"]
        cash_ratio = self.cash / 1000000  # Saldo normalizado [0, 1]

        return np.array([price, demand, idh, crime, infra, cash_ratio], dtype=np.float32)
###################################################################################################################
    
    def _calculate_property_value(self):
        """Calcula o valor total dos imóveis comprados com base no preço atualizado de mercado."""
        total_property_value = sum(prop["preco"] * np.random.uniform(0.9, 1.3) for prop in self.owned_properties)
        return total_property_value
    
###################################################################################################################
    
    def step(self, action):
        """Executa uma ação no ambiente e retorna (novo estado, recompensa, done, info)."""
        if self.current_step >= len(self.market) - 1:
            return self._get_observation(), 0, True, {}
        reward = 0
        done = False
        prop = self.market[self.current_step]
        price = prop["preco"]
    
        # Variável para contar imóveis vendidos no episódio
        if not hasattr(self, 'num_vendidos_step'):
            self.num_vendidos_step = 0
    
        #  Se o agente ficou esperando por mais de 20 episódios, força uma compra 
        if self.waiting_steps >= 20:
                action = 0  # Força a compra
    
        # 🏠 Número de imóveis antes da ação
        previous_owned_count = len(self.owned_properties)
    
        if action == 0:  # Comprar
            if self.cash >= price:
                self.owned_properties.append(prop)
                self.cash -= price
                reward = 1 + (200000 - price) / 50000  
                self.waiting_steps = 0  # Reseta o contador de espera
    
        elif action == 2 and len(self.owned_properties) > 0:  # Vender
            property_data = self.owned_properties.pop(0)
            sell_price = property_data["preco"] * np.random.uniform(0.7, 1.5)
    
            if property_data.get("tempo_no_mercado", 0) > 10:
                sell_price *= 0.9  
    
            reward = (sell_price - property_data["preco"]) / 10000
            self.cash += sell_price
            self.waiting_steps = 0  # Reseta o contador de espera
    
        elif action == 1:  # Esperar
            self.waiting_steps += 1  # Incrementa contador de espera
        
        if self.current_step % 10 == 0:
            self._apply_market_events()
    
        self.current_step += 1
        return self._get_observation(), reward, done, {}

    
###################################################################################################################
    
    def render_grafs(self):
        """Renderiza o ambiente visualmente usando matplotlib."""
        if self.render_mode == 'human':
            profit = self.cash - self.initial_cash  # Lucro
            total_property_value = self._calculate_property_value()
            patrimonio_total = self.cash + total_property_value  # Patrimônio = Dinheiro + Valor dos imóveis
            total_imoveis = len(self.owned_properties)  # Total de imóveis comprados
            waitstep = self.waiting_steps  # Contador de espera
            self.history.append((self.current_step, self.cash, patrimonio_total, total_imoveis, waitstep))
            print(f"Passo {self.current_step} | Saldo: R${self.cash:.2f} | Imóveis: {total_imoveis} | Lucro: R${profit:.2f} | Patrimônio: R${patrimonio_total:.2f} | Esperando: {waitstep} passos")
            if len(self.history) < 2:
                return
            if self.fig is None or self.axs is None:
                plt.ion()
                self.fig, self.axs = plt.subplots(2, 2, figsize=(12, 8))
            self.axs[0, 0].clear()
            self.axs[0, 1].clear()
            self.axs[1, 0].clear()
            self.axs[1, 1].clear()
            steps, cash_values, patrimonio_values, imoveis_comprados, waitstep = zip(*self.history)
            # 📈 Gráfico de saldo disponível
            self.axs[0, 0].plot(steps, cash_values, label="Saldo Disponível", color="blue")
            self.axs[0, 0].set_title("Saldo Disponível")
            self.axs[0, 0].grid(True)
            # 📈 Gráfico de patrimônio total
            self.axs[0, 1].plot(steps, patrimonio_values, label="Patrimônio Total", color="green")
            self.axs[0, 1].set_title("Patrimônio Total")
            self.axs[0, 1].grid(True)
            # 📊 Gráfico de número de imóveis comprados
            self.axs[1, 0].plot(steps, imoveis_comprados, color="orange", label="Imóveis Comprados")
            self.axs[1, 0].set_title("Número de Imóveis Comprados")
            self.axs[1, 0].grid(True)
            # 📊 Gráfico de contador de espera
            self.axs[1, 1].plot(steps, waitstep, color="red", label="Imóveis Vendidos no Episódio")
            self.axs[1, 1].set_title("Contador de Espera")
            self.axs[1, 1].grid(True)

            plt.tight_layout()
            plt.pause(0.05)

###################################################################################################################
    
    def reset(self):
        """Reseta o ambiente para um novo episódio."""
        self.cash = 100000
        self.owned_properties = []
        self.current_step = 0
        #self.market = self._generate_market()
        return self._get_observation()

###################################################################################################################

    def _mapear_bairros_para_poligonos(self, distritos):
        """Associa cada bairro ao polígono correspondente com base no nome."""
        mapa = {}
        for d in distritos:
            nome = d["nome"].strip().upper()
            mapa[nome] = d["poligono"]
        return mapa
    
    
    def ponto_aleatorio_em_poligono(self, poligono, tentativas=100):
        from pathlib import Path
        output_path = Path("environments/GEO/tests/imoveis_fixos.json")
        poly = Polygon(poligono)
        minx, miny, maxx, maxy = poly.bounds
        for _ in range(tentativas):
            x = random.uniform(minx, maxx)
            y = random.uniform(miny, maxy)
            if poly.contains(Point(x, y)):
                return (x, y)
        return ((minx + maxx) / 2, (miny + maxy) / 2)

###################################################################################################################
        
    def render_folium_timelapse_v1(self, historico, save_path="environments/GEO/tests/mapa_animado_v1.html"):
        m = folium.Map(location=[-23.55, -46.63], zoom_start=9, tiles="CartoDB positron")

        # Plota outline da cidade
        from environments.GEO.maps.SP import distritos
        for d in distritos:
            folium.Polygon(
                locations=[(lat, lon) for lon, lat in d["poligono"]],
                color="black", weight=2, fill=True, fill_opacity=0.01
            ).add_to(m)

        # Dicionário base com todos os imóveis e tempo inicial
        features = {}
        for prop in self.market:
            lon, lat = prop["pos"]
            id_key = f"{lon:.6f}_{lat:.6f}"  # chave única
            features[id_key] = {
                "prop": prop,
                "color": "lightgrey",  # todos começam pretos
                "step": 0
            }

        # Atualiza status conforme histórico
        for h in historico:
            prop = h["prop"]
            lon, lat = prop["pos"]
            id_key = f"{lon:.6f}_{lat:.6f}"
            if id_key not in features:
                continue
            if h["sold"]:
                features[id_key]["color"] = "cyan"
                features[id_key]["step"] = h["step"]
            elif h["owned"]:
                if features[id_key]["color"] != "cyan":  # só muda se não foi vendido
                    features[id_key]["color"] = "red"
                    features[id_key]["step"] = h["step"]

        # Monta o GeoJSON
        geojson_features = []
        for id_key, data in features.items():
            prop = data["prop"]
            lon, lat = prop["pos"]
            popup_html = (
                f"<b>{prop['tipo']}</b><br>"
                f"Bairro: {prop['bairro']}<br>"
                f"Preço: R${prop['preco']:,}<br>"
                f"Metragem: {prop['metragem']} m²<br>"
                f"IDH: {prop['idh_microrregiao']:.3f}<br>"
                f"Criminalidade: {prop['taxa_criminalidade']:.2f}<br>"
                f"Infraestrutura: {prop['infraestrutura']:.2f}"
            )

            geojson_features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat],
                },
                "properties": {
                    "time": (datetime.datetime(2023, 1, 1) + datetime.timedelta(seconds=data["step"])).isoformat(),
                    "style": {"color": data["color"], "fillColor": data["color"], "fillOpacity": 1, "radius": 1},
                    "icon": "circle",
                    "popup": popup_html
                }
            })

        TimestampedGeoJson({
            "type": "FeatureCollection",
            "features": geojson_features
        }, period="PT1S", add_last_point=True, auto_play=True, loop=False, max_speed=2).add_to(m)

        m.save(save_path)
        print(f"✅ Mapa dinâmico salvo em: {save_path}")

###################################################################################################################





































###################################################################################################################



