

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pandas as pd
import matplotlib.pyplot as plt
import time
import pygame
###################################################################################################################

class HomeChoiceEnv(gym.Env):
    """
    Simulador de investimento imobiliário na cidade de São Paulo.
    O agente deve comprar e vender imóveis para atingir R$ 1.000.000.
    O mercado é dinâmico, com valorização e desvalorização dos imóveis baseada em características reais.
    """
    def __init__(self, render_mode='human'):
        super().__init__()
        self.render_mode        = render_mode
        self.history            = []  # Histórico para renderização gráfica
        self.fig, self.ax       = None, None
        self.initial_cash       = 100000 # Saldo inicial do agente
        self.cash               = 100000 
        self.owned_properties   = []  # Lista de imóveis comprados
        self.current_step       = 0
        self.waiting_steps      = 0 
        self.action_space       = spaces.Discrete(3) # Espaço de Ação: 0 = Comprar, 1 = Esperar, 2 = Vender
        self.observation_space  = spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32) # Espaço de Observação: [preço do imóvel, metragem, IDH, taxa de criminalidade, infraestrutura, saldo do agente]
        self.idh_bairros        = {
            "Pinheiros": 0.942, "Sé": 0.889, "Lapa": 0.906, "Mooca": 0.869,
            "Santana / Tucuruvi": 0.869, "Vila Mariana": 0.938, "Butantã": 0.859,
            "Ipiranga": 0.824, "Jabaquara": 0.816, "Santo Amaro": 0.909,
            "Casa Verde": 0.799, "Penha": 0.804, "São Miguel Paulista": 0.736,
            "Perus": 0.731, "Cidade Ademar": 0.758, "Freguesia do Ó / Brasilândia": 0.762,
            "Pirituba / Jaraguá": 0.787, "Campo Limpo": 0.783, "Ermelino Matarazzo": 0.777,
            "M’Boi Mirim": 0.716, "Cidade Tiradentes": 0.708, "Guaianases": 0.713,
            "São Mateus": 0.732, "Itaim Paulista": 0.725, "Itaquera": 0.758
        }
        self.market             = self._generate_market()

###################################################################################################################
    def _generate_market(self):
        """Gera um bairro fictício de imóveis com características variadas, refletindo a realidade de São Paulo."""
        market  = []
        bairros = list(self.idh_bairros.keys())
        for _ in range(100000):  # 100.000 imóveis
            bairro = np.random.choice(bairros)
            idh = self.idh_bairros.get(bairro, 0.8)  # Se não encontrar, assume IDH médio de 0.8
            # Ajuste da distribuição de tipos de imóveis conforme o IDH do bairro
            if idh > 0.85:
                tipo_imovel = np.random.choice(
                    ["Apartamento Padrão", "Casa de Luxo", "Cobertura"],
                    p=[0.5, 0.3, 0.2]
                )  # Mais apartamentos e coberturas em bairros ricos
            elif idh > 0.75:
                tipo_imovel = np.random.choice(
                    ["Casa Popular", "Apartamento Padrão", "Casa de Luxo"],
                    p=[0.3, 0.5, 0.2]
                )  # Predomínio de apartamentos padrão e algumas casas de luxo
            else:
                tipo_imovel = np.random.choice(
                    ["Casa Popular", "Apartamento Padrão"],
                    p=[0.7, 0.3]
                )  # Mais casas populares em bairros menos desenvolvidos

            # Definição do preço médio do metro quadrado conforme o IDH
            preco_m2_base = np.interp(idh, [0.7, 0.95], [2000, 15000])  # Entre R$ 1.000 e R$ 15.000/m²

            # Ajuste das características do imóvel conforme o tipo
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
            else:  # Cobertura
                metragem = np.random.randint(150, 401)
                preco = int(metragem * preco_m2_base * np.random.uniform(1.2, 1.5))
                condominio = np.random.randint(2000, 5001)

            # Infraestrutura aumenta em bairros mais ricos
            infraestrutura = np.interp(idh, [0.7, 0.95], [0.3, 1.0])  # Infraestrutura de 0.3 a 1.0

            # Taxa de criminalidade diminui em bairros mais ricos
            taxa_criminalidade = np.interp(idh, [0.7, 0.95], [1.0, 0.2])  # Criminalidade de 1.0 a 0.2

            # Demanda do mercado varia conforme a atratividade do bairro
            demanda = int(np.interp(idh, [0.7, 0.95], [300, 1000]) * np.random.uniform(0.8, 1.2))

            # Criando o imóvel
            property_data = {
                "tipo": tipo_imovel,
                "bairro": bairro,
                "idh_microrregiao": idh,
                "metragem": metragem,
                "preco": preco,
                "condominio": condominio,
                "taxa_criminalidade": taxa_criminalidade,
                "infraestrutura": infraestrutura,
                "demanda": demanda,
                "tempo_no_mercado": 0
            }
            market.append(property_data)

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
        self.market = self._generate_market()
        return self._get_observation()

###################################################################################################################

    def render_pygame_v0(self):
        import pygame

        if not pygame.get_init():
            pygame.init()

        # Inicializa tela, fonte e clock apenas uma vez
        if not hasattr(self, "screen"):
            self.screen_width, self.screen_height = 800, 600
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("🏡 Real Estate RL Simulator")
            self.font = pygame.font.SysFont("Arial", 18)
            self.clock = pygame.time.Clock()

        self.screen.fill((240, 240, 240))

        if self.current_step >= len(self.market):
            return

        prop = self.market[self.current_step]
        tipo = prop["tipo"]
        bairro = prop["bairro"]
        preco = prop["preco"]
        metragem = prop["metragem"]
        idh = prop["idh_microrregiao"]
        crime = prop["taxa_criminalidade"]
        infra = prop["infraestrutura"]

        tipo_colors = {
            "Casa Popular": (100, 149, 237),        # Azul
            "Apartamento Padrão": (60, 179, 113),   # Verde
            "Casa de Luxo": (255, 215, 0),          # Dourado
            "Cobertura": (138, 43, 226)             # Roxo
        }
        color = tipo_colors.get(tipo, (200, 200, 200))

        # Desenha o imóvel como um quadrado no centro
        pygame.draw.rect(self.screen, color, pygame.Rect(350, 200, 100, 100))
        pygame.draw.rect(self.screen, (0, 0, 0), pygame.Rect(350, 200, 100, 100), 2)

        patrimonio = self.cash + self._calculate_property_value()
        textos = [
            f"🏘️ Tipo: {tipo}",
            f"📍 Bairro: {bairro}",
            f"💰 Preço: R${preco:,.0f}",
            f"📐 Metragem: {metragem}m²",
            f"🔢 IDH: {idh:.3f}",
            f"🚨 Criminalidade: {crime:.2f}",
            f"🏗️ Infraestrutura: {infra:.2f}",
            f"💵 Saldo: R${self.cash:,.0f}",
            f"📦 Imóveis: {len(self.owned_properties)}",
            f"🧮 Patrimônio: R${patrimonio:,.0f}",
            f"⏳ Espera: {self.waiting_steps}"
        ]

        for i, texto in enumerate(textos):
            rendered = self.font.render(texto, True, (0, 0, 0))
            self.screen.blit(rendered, (20, 20 + i * 25))

        pygame.display.flip()
        self.clock.tick(60)

###################################################################################################################
    def close_pygame(self):
        pygame.quit()
###################################################################################################################










































###################################################################################################################



