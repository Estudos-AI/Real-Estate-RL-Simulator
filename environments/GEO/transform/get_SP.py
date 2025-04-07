# SÃO PAULO - MAPA DE DISTRITOS
# Este script gera um mapa de contorno dos distritos da cidade de São Paulo, a partir de um arquivo GeoJSON.
# Permite visualizar o mapa antes de exportar para coordenadas pygame.

import geopandas as gpd
import matplotlib.pyplot as plt
import os



# Lista oficial dos nomes dos distritos de São Paulo (ordem igual à do GeoJSON)
nomes_distritos = [
    "PIRITUBA", "SAO DOMINGOS", "JARAGUA", "BRASILANDIA", "FREGUESIA DO O", "CASA VERDE", "CACHOEIRINHA", "LIMAO",
    "VILA GUILHERME", "VILA MARIA", "VILA MEDEIROS", "ARTUR ALVIM", "PENHA", "CANGAIBA", "VILA MATILDE", "PONTE RASA",
    "ERMELINO MATARAZZO", "VILA CURUCA", "ITAIM PAULISTA", "GUAIANASES", "LAJEADO", "BARRA FUNDA", "PERDIZES",
    "VILA LEOPOLDINA", "JAGUARA", "LAPA", "JAGUARE", "REPUBLICA", "SANTA CECILIA", "SE", "BELA VISTA", "BOM RETIRO",
    "CAMBUCI", "CONSOLACAO", "LIBERDADE", "MOOCA", "PARI", "TATUAPE", "AGUA RASA", "BELEM", "BRAS", "CARRAO",
    "VILA FORMOSA", "ARICANDUVA", "SAO MATEUS", "SAO RAFAEL", "IGUATEMI", "VILA PRUDENTE", "SAO LUCAS", "MORUMBI",
    "RIO PEQUENO", "VILA SONIA", "BUTANTA", "RAPOSO TAVARES", "PINHEIROS", "ALTO DE PINHEIROS", "ITAIM BIBI",
    "JARDIM PAULISTA", "CAMPO LIMPO", "CAPAO REDONDO", "VILA ANDRADE", "JARDIM ANGELA", "JARDIM SAO LUIS",
    "SOCORRO", "CIDADE DUTRA", "GRAJAU", "MARSILAC", "PARELHEIROS", "CIDADE TIRADENTES", "PERUS", "ANHANGUERA",
    "SAPOPEMBA", "SACOMA", "CURSINO", "IPIRANGA", "MOEMA", "SAUDE", "VILA MARIANA", "PEDREIRA", "CIDADE ADEMAR",
    "JACANA", "TREMEMBE", "MANDAQUI", "SANTANA", "TUCURUVI", "SANTO AMARO", "CAMPO GRANDE", "CAMPO BELO",
    "JABAQUARA", "VILA JACUI", "SAO MIGUEL", "JARDIM HELENA", "CIDADE LIDER", "PARQUE DO CARMO", "JOSE BONIFACIO",
    "ITAQUERA"
]


# Caminhos
geojson_path = os.path.join("environments", "GEO", "raw" , "distritos.geojson")
output_path  = os.path.join("environments", "GEO", "maps", "SP.py")

# Parâmetros ajustáveis
scale = 0.00005         # quanto menor, mais cabe na tela
offset = (400, 550)     # move o mapa no plano (x, y)

# Carrega o GeoJSON
gdf = gpd.read_file(geojson_path)
gdf = gdf.to_crs(epsg=3857)  # Reprojeta para metros


def geo_to_pygame_coords(geometry, scale, offset):
    if geometry.geom_type == "Polygon":
        return [[(x * scale + offset[0], -y * scale + offset[1]) for x, y in geometry.exterior.coords]]
    elif geometry.geom_type == "MultiPolygon":
        return [
            [(x * scale + offset[0], -y * scale + offset[1]) for x, y in poly.exterior.coords]
            for poly in geometry.geoms
        ]
    return []



# Constrói lista com nome e polígono
distritos_py = []
for idx, row in gdf.iterrows():
    try:
        nome = nomes_distritos[idx]
    except IndexError:
        nome = f"Distrito_{idx}"

    polys = geo_to_pygame_coords(row.geometry, scale, offset)
    for poly in polys:
        distritos_py.append({"nome": nome, "poligono": poly})

# Visualização
plt.figure(figsize=(8, 6))
for d in distritos_py:
    xs, ys = zip(*d["poligono"])
    plt.plot(xs, ys, color="black", linewidth=0.6)
plt.title("Pré-visualização dos distritos (ajuste scale/offset se necessário)")
plt.gca().invert_yaxis()
plt.axis("equal")
plt.tight_layout()
# plt.show()


plt.savefig("images/mapa_distritos.png", dpi=300)
plt.close()
print("✅ Mapa salvo em 'images/mapa_distritos.png'")


# Exporta para SP.py
with open(output_path, "w", encoding="utf-8") as f:
    f.write("# Contornos dos distritos de São Paulo com nomes e coordenadas pygame\n")
    f.write("distritos = [\n")
    for d in distritos_py:
        f.write(f"    {{'nome': '{d['nome']}', 'poligono': {d['poligono']}}},\n")
    f.write("]\n")

print(f"✅ {len(distritos_py)} polígonos com nome salvos em '{output_path}'")
print(f"📐 Scale usado: {scale}")
print(f"🧭 Offset usado: {offset}")




# # Converte todos os distritos
# pygame_polygons = []
# for geom in gdf.geometry:
#     polygons = geo_to_pygame_coords(geom, scale=scale, offset=offset)
#     pygame_polygons.extend(polygons)

# # Visualização com matplotlib
# plt.figure(figsize=(8, 6))
# for poly in pygame_polygons:
#     xs, ys = zip(*poly)
#     plt.plot(xs, ys, color="black", linewidth=0.7)
# plt.title("Pré-visualização do mapa vetorial (ajuste scale e offset se necessário)")
# plt.gca().invert_yaxis()
# plt.axis("equal")
# plt.tight_layout()
# plt.show()

# # Exporta como arquivo .py com lista distritos
# with open(output_path, "w", encoding="utf-8") as f:
#     f.write("# Contornos dos distritos de São Paulo em coordenadas pygame\n")
#     f.write("distritos = [\n")
#     for poly in pygame_polygons:
#         f.write("    " + str(poly) + ",\n")
#     f.write("]\n")

# print(f"✅ Contornos salvos em '{output_path}' com {len(pygame_polygons)} polígonos.")
# print(f"📐 Scale usado: {scale}")
# print(f"🧭 Offset usado: {offset}")
