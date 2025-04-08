# SÃO PAULO - MAPA DE DISTRITOS (em coordenadas reais para mapas interativos)

import geopandas as gpd
import matplotlib.pyplot as plt
import os

# Caminhos
geojson_path = os.path.join("environments", "GEO", "raw", "distritos.geojson")
output_path  = os.path.join("environments", "GEO", "maps", "SP.py")

# Carrega o GeoJSON e converte para WGS84 (lat/lon)
gdf = gpd.read_file(geojson_path)
gdf = gdf.to_crs(epsg=4326)  # EPSG 4326 = latitude/longitude padrão

# Lista oficial dos nomes (ordem igual à do GeoJSON)
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

# Função para extrair coordenadas geográficas
def geo_to_latlon_coords(geometry):
    if geometry.geom_type == "Polygon":
        return [list(geometry.exterior.coords)]
    elif geometry.geom_type == "MultiPolygon":
        return [list(poly.exterior.coords) for poly in geometry.geoms]
    return []

# Constrói lista com nome e polígono
distritos_py = []
for idx, row in gdf.iterrows():
    nome = nomes_distritos[idx] if idx < len(nomes_distritos) else f"Distrito_{idx}"
    polys = geo_to_latlon_coords(row.geometry)
    for poly in polys:
        distritos_py.append({"nome": nome, "poligono": poly})

# Visualização com matplotlib
plt.figure(figsize=(8, 6))
for d in distritos_py:
    xs, ys = zip(*d["poligono"])
    plt.plot(xs, ys, color="black", linewidth=0.7)
plt.title("Mapa dos Distritos de São Paulo (EPSG:4326)")
#plt.gca().invert_yaxis()
plt.axis("equal")
plt.tight_layout()
plt.savefig("images/mapa_SP_bairros.png", dpi=300)
plt.close()
print("✅ Mapa salvo em 'images/mapa_SP_bairros.png'")

# Exporta para SP.py (coordenadas reais)
with open(output_path, "w", encoding="utf-8") as f:
    f.write("# Contornos dos distritos de São Paulo em coordenadas geográficas (lat/lon)\n")
    f.write("distritos = [\n")
    for d in distritos_py:
        f.write(f"    {{'nome': '{d['nome']}', 'poligono': {d['poligono']}}},\n")
    f.write("]\n")

print(f"✅ {len(distritos_py)} polígonos com nome salvos em '{output_path}'")
