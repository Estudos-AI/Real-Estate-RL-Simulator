# def render_pygame_v0(self):
#         import pygame

#         if not pygame.get_init():
#             pygame.init()

#         # Inicializa tela, fonte e clock apenas uma vez
#         if not hasattr(self, "screen"):
#             self.screen_width, self.screen_height = 800, 600
#             self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
#             pygame.display.set_caption("🏡 Real Estate RL Simulator")
#             self.font = pygame.font.SysFont("Arial", 18)
#             self.clock = pygame.time.Clock()

#         self.screen.fill((240, 240, 240))

#         if self.current_step >= len(self.market):
#             return

#         prop = self.market[self.current_step]
#         tipo = prop["tipo"]
#         bairro = prop["bairro"]
#         preco = prop["preco"]
#         metragem = prop["metragem"]
#         idh = prop["idh_microrregiao"]
#         crime = prop["taxa_criminalidade"]
#         infra = prop["infraestrutura"]

#         tipo_colors = {
#             "Casa Popular": (100, 149, 237),        # Azul
#             "Apartamento Padrão": (60, 179, 113),   # Verde
#             "Casa de Luxo": (255, 215, 0),          # Dourado
#             "Cobertura": (138, 43, 226)             # Roxo
#         }
#         color = tipo_colors.get(tipo, (200, 200, 200))

#         # Desenha o imóvel como um quadrado no centro
#         pygame.draw.rect(self.screen, color, pygame.Rect(350, 200, 100, 100))
#         pygame.draw.rect(self.screen, (0, 0, 0), pygame.Rect(350, 200, 100, 100), 2)

#         patrimonio = self.cash + self._calculate_property_value()
#         textos = [
#             f"🏘️ Tipo: {tipo}",
#             f"📍 Bairro: {bairro}",
#             f"💰 Preço: R${preco:,.0f}",
#             f"📐 Metragem: {metragem}m²",
#             f"🔢 IDH: {idh:.3f}",
#             f"🚨 Criminalidade: {crime:.2f}",
#             f"🏗️ Infraestrutura: {infra:.2f}",
#             f"💵 Saldo: R${self.cash:,.0f}",
#             f"📦 Imóveis: {len(self.owned_properties)}",
#             f"🧮 Patrimônio: R${patrimonio:,.0f}",
#             f"⏳ Espera: {self.waiting_steps}"
#         ]

#         for i, texto in enumerate(textos):
#             rendered = self.font.render(texto, True, (0, 0, 0))
#             self.screen.blit(rendered, (20, 20 + i * 25))

#         pygame.display.flip()
#         self.clock.tick(60)

# ###################################################################################################################
#     def render_pygame_v2(self):
#         import pygame
#         from environments.GEO.maps.SP import distritos

#         # Inicializa pygame
#         if not pygame.get_init():
#             pygame.init()

#         # Inicializa tela e assets uma única vez
#         if not hasattr(self, "screen"):
#             self.screen_width, self.screen_height = 800, 600
#             self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
#             pygame.display.set_caption("🏡 Real Estate RL Simulator")
#             self.font = pygame.font.SysFont("Arial", 18)
#             self.clock = pygame.time.Clock()

#             # Mapa desenhado apenas uma vez
#             self.mapa_surface = pygame.Surface((self.screen_width, self.screen_height))
#             self.mapa_surface.fill((240, 240, 240))

#             for d in distritos:
#                 poly = d.get("poligono")
#                 if not isinstance(poly, list) or not all(isinstance(p, tuple) and len(p) == 2 for p in poly):
#                     continue
#                 try:
#                     pygame.draw.polygon(self.mapa_surface, (0, 0, 0), poly, width=1)
#                 except:
#                     continue

#         # Cola o fundo do mapa
#         self.screen.blit(self.mapa_surface, (0, 0))

#         # Desenha os imóveis
#         for prop in self.market:
#             pos = prop.get("pos")
#             if not pos:
#                 continue

#             if prop in self.owned_properties:
#                 color = (60, 179, 113)  # Verde
#             elif hasattr(self, "vendidos") and prop in self.vendidos:
#                 color = (220, 20, 60)  # Vermelho
#             elif prop == self.market[self.current_step]:
#                 color = (255, 215, 0)  # Amarelo
#             else:
#                 color = (100, 149, 237)  # Azul

#             pygame.draw.circle(self.screen, color, pos, 4)

#         # # Desenha HUD
#         patrimonio = self.cash + self._calculate_property_value()
#         hud = [
#             f"Passo: {self.current_step}",
#             f"Saldo: R${self.cash:,.0f}",
#             f"Imóveis: {len(self.owned_properties)}",
#             f"Patrimônio: R${patrimonio:,.0f}"
#         ]

#         for i, text in enumerate(hud):
#             rendered = self.font.render(text, True, (0, 0, 0))
#             self.screen.blit(rendered, (10, 10 + i * 22))

#         pygame.display.flip()
#         self.clock.tick(60)

        
# ###################################################################################################################
#     def render_pygame_v3(self):
#         import pygame
#         import os
#         from environments.GEO.maps.SP import distritos
#         import matplotlib.pyplot as plt

#         mapa_path = "images/mapa_SP_bairros_v3.png"

#         # Gera e salva o mapa apenas uma vez usando matplotlib
#         if not os.path.exists(mapa_path):
#             print("🗺️  Gerando imagem do mapa base...")
#             plt.figure(figsize=(8, 6))
#             for d in distritos:
#                 if "poligono" in d and isinstance(d["poligono"], list):
#                     xs, ys = zip(*d["poligono"])
#                     plt.plot(xs, ys, color="black", linewidth=0.5)
#             plt.title("Mapa dos Distritos de SP")
#             plt.gca().invert_yaxis()
#             plt.axis("equal")
#             plt.axis("off")
#             plt.tight_layout()
#             plt.savefig(mapa_path, dpi=150)
#             plt.close()

#         # Inicializa pygame
#         if not pygame.get_init():
#             pygame.init()

#         if not hasattr(self, "screen"):
#             self.screen_width, self.screen_height = 800, 600
#             self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
#             pygame.display.set_caption("🏡 Real Estate RL Simulator")
#             self.font = pygame.font.SysFont("Arial", 18)
#             self.clock = pygame.time.Clock()

#             self.mapa_surface = pygame.image.load(mapa_path).convert()

#         # Cola o fundo do mapa
#         self.screen.blit(self.mapa_surface, (0, 0))

#         # Desenha os imóveis
#         for prop in self.market:
#             pos = prop.get("pos")
#             if not pos:
#                 continue

#             if prop in self.owned_properties:
#                 color = (60, 179, 113)  # Verde
#             elif hasattr(self, "vendidos") and prop in self.vendidos:
#                 color = (220, 20, 60)  # Vermelho
#             elif prop == self.market[self.current_step]:
#                 color = (255, 215, 0)  # Amarelo
#             else:
#                 color = (100, 149, 237)  # Azul

#             pygame.draw.circle(self.screen, color, pos, 4)

#         # HUD
#         patrimonio = self.cash + self._calculate_property_value()
#         hud = [
#             f"Passo: {self.current_step}",
#             f"Saldo: R${self.cash:,.0f}",
#             f"Imóveis: {len(self.owned_properties)}",
#             f"Patrimônio: R${patrimonio:,.0f}"
#         ]
#         for i, text in enumerate(hud):
#             rendered = self.font.render(text, True, (0, 0, 0))
#             self.screen.blit(rendered, (10, 10 + i * 22))

#         pygame.display.flip()
#         self.clock.tick(60)
        
# ###################################################################################################################
#     def close_pygame(self):
#         pygame.quit()
###################################################################################################################
#   def render_geemap_folium_v1(self, save_path="mapa_interativo.html"):
#         import folium
#         import geemap.foliumap as geemap

#         # Mapa base centralizado em São Paulo
#         m = geemap.Map(center=[-23.55, -46.63], zoom=11)

#         # Desenha cada bairro com cor suave e popup informativo
#         for nome, poligono in self.bairro_poligonos.items():
#             if not poligono:
#                 continue

#             # Conversão pygame → latitude/longitude precisa ter sido feita ANTES
#             coords = [(lat, lon) for lon, lat in poligono]  # Inverter (x, y) → (lat, lon)

#             media_idh = self.idh_bairros.get(nome.upper(), 0.8)
#             imoveis_bairro = [p for p in self.market if p["bairro"].upper() == nome.upper()]
#             media_preco = np.mean([p["preco"] for p in imoveis_bairro]) if imoveis_bairro else 0
#             n_imoveis = len(imoveis_bairro)

#             popup_html = f"""
#             <b>{nome.title()}</b><br>
#             🧮 IDH médio: {media_idh:.3f}<br>
#             🏠 Imóveis: {n_imoveis}<br>
#             💰 Preço médio: R${media_preco:,.0f}
#             """

#             folium.Polygon(
#                 locations=coords,
#                 color="black",
#                 fill=True,
#                 fill_opacity=0.07,
#                 weight=1,
#                 popup=folium.Popup(popup_html, max_width=250)
#             ).add_to(m)

#         # Adiciona os imóveis como pontos no mapa com cores diferentes por status
#         for prop in self.market:
#             latlon = (prop["pos"][1], prop["pos"][0])  # (y, x) → (lat, lon)

#             if prop in self.owned_properties:
#                 color = "green"
#             elif hasattr(self, "vendidos") and prop in self.vendidos:
#                 color = "red"
#             elif prop == self.market[self.current_step]:
#                 color = "orange"
#             else:
#                 color = "blue"

#             popup = folium.Popup(
#                 f"""
#                 <b>{prop["tipo"]}</b><br>
#                 📍 Bairro: {prop["bairro"]}<br>
#                 💰 Preço: R${prop["preco"]:,.0f}<br>
#                 📐 Metragem: {prop["metragem"]} m²<br>
#                 🔢 IDH: {prop["idh_microrregiao"]:.3f}<br>
#                 🚨 Crime: {prop["taxa_criminalidade"]:.2f}<br>
#                 🏗️ Infra: {prop["infraestrutura"]:.2f}
#                 """,
#                 max_width=300
#             )

#             folium.CircleMarker(
#                 location=latlon,
#                 radius=3.5,
#                 color=color,
#                 fill=True,
#                 fill_color=color,
#                 fill_opacity=0.9,
#                 popup=popup
#             ).add_to(m)

#         # HUD: pode ser só um título ou render em HTML depois
#         folium.Marker(
#             location=[-23.3, -46.9],
#             icon=folium.DivIcon(html=f"""
#             <div style="font-size: 16px; font-weight: bold">
#             💵 Saldo: R${self.cash:,.0f} | 🏠 Imóveis: {len(self.owned_properties)} | 
#             🧮 Patrimônio: R${self.cash + self._calculate_property_value():,.0f}
#             </div>
#             """)
#         ).add_to(m)

#         # Salva o HTML interativo
#         m.save(save_path)
#         print(f"✅ Mapa salvo em: {save_path}")

# ###################################################################################################################]

#     def render_folium_map_v2(self, save_path="mapa.html"):
#         # Mapa base
#         mapa = folium.Map(location=[-23.55, -46.63], zoom_start=11, tiles="cartodbpositron")
    
#         # Adiciona contorno dos distritos (GeoJSON)
#         import geopandas as gpd
#         gdf = gpd.read_file("environments/GEO/raw/distritos.geojson")
#         folium.GeoJson(gdf, name="Distritos").add_to(mapa)
    
#         # Cluster para performance
#         cluster = MarkerCluster().add_to(mapa)
    
#         for i, prop in enumerate(self.market):
#             lat, lon = prop["pos"][1], prop["pos"][0]
#             point = [lat, lon]
    
#             # Define cor
#             if prop in self.owned_properties:
#                 color = "green"
#             elif hasattr(self, "vendidos") and prop in self.vendidos:
#                 color = "red"
#             elif i == self.current_step:
#                 color = "orange"
#             else:
#                 color = "blue"
    
#             # Texto do popup
#             popup = folium.Popup(f"""
#             <b>{prop['tipo']}</b><br>
#             Bairro: {prop['bairro']}<br>
#             Preço: R${prop['preco']:,.0f}<br>
#             Metragem: {prop['metragem']} m²<br>
#             IDH: {prop['idh_microrregiao']:.3f}<br>
#             Criminalidade: {prop['taxa_criminalidade']:.2f}<br>
#             Infraestrutura: {prop['infraestrutura']:.2f}<br>
#             """, max_width=300)
    
#             folium.CircleMarker(
#                 location=point,
#                 radius=5,
#                 color=color,
#                 fill=True,
#                 fill_color=color,
#                 fill_opacity=0.7,
#                 popup=popup
#             ).add_to(cluster)
    
#         # HUD
#         patrimonio = self.cash + self._calculate_property_value()
#         folium.Marker(
#             location=[-23.30, -46.95],
#             icon=folium.DivIcon(html=f"""
#             <div style="font-family:Arial; background:white; padding:10px; border:2px solid gray; border-radius:8px;">
#                 <b>Passo:</b> {self.current_step}<br>
#                 <b>Saldo:</b> R${self.cash:,.0f}<br>
#                 <b>Imóveis:</b> {len(self.owned_properties)}<br>
#                 <b>Patrimônio:</b> R${patrimonio:,.0f}
#             </div>""")
#         ).add_to(mapa)
    
#         # Salva
#         mapa.save(save_path)
#         print(f"🗺️  Mapa salvo como '{save_path}'")
        
        
#     def render_folium_timelapse_v0(self, historico, save_path="environments/GEO/tests/mapa_animado_v0.html"):
#         m = folium.Map(location=[-23.55, -46.63], zoom_start=11, tiles="CartoDB positron")
#         try:
#             gdf = gpd.read_file("environments/GEO/raw/distritos.geojson")
#             folium.GeoJson(
#                 gdf,
#                 name="Distritos",
#                 style_function=lambda x: {
#                     "fillColor": "#ffffff",  
#                     "color": "#000000",       
#                     "weight": 1.2
#                 }
#             ).add_to(m)
#         except Exception as e:
#             print(f"⚠️ Erro ao carregar contorno dos distritos: {e}")
#         features = []
#         for h in historico:
#             prop = h["prop"]
#             if "pos" not in prop:
#                 continue
#             lon, lat = prop["pos"]
#             cor = "black"
#             if h["owned"]:
#                 cor = "red"
#             elif h["sold"]:
#                 cor = "cyan"
#             # elif h["step"] == self.current_step:
#             #     cor = "orange"

#             popup_html = (
#                 f"<b>{prop['tipo']}</b><br>"
#                 f"Bairro: {prop['bairro']}<br>"
#                 f"Preço: R${prop['preco']:,}<br>"
#                 f"Metragem: {prop['metragem']} m²<br>"
#                 f"IDH: {prop['idh_microrregiao']:.3f}<br>"
#                 f"Criminalidade: {prop['taxa_criminalidade']:.2f}<br>"
#                 f"Infraestrutura: {prop['infraestrutura']:.2f}"
#             )

#             feature = {
#                 "type": "Feature",
#                 "geometry": {
#                     "type": "Point",
#                     "coordinates": [lon, lat],
#                 },
#                 "properties": {
#                     "time": (datetime.datetime(2023, 1, 1) + datetime.timedelta(seconds=h["step"])).isoformat(),
#                     "style": {
#                         "color": cor,
#                         "fillColor": cor,
#                         "opacity": 0.7,
#                         "fillOpacity": 0.4,
#                         "radius": 4
#                     },
#                     "icon": "circle",
#                     "popup": popup_html
#                 }
#             }
#             features.append(feature)

#         TimestampedGeoJson({
#             "type": "FeatureCollection",
#             "features": features
#         }, period="PT1S", add_last_point=True, auto_play=True, loop=False, max_speed=2).add_to(m)

#         m.save(save_path)
#         print(f"✅ Mapa dinâmico salvo em: {save_path}")
        