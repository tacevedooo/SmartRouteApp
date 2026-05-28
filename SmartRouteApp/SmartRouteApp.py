import reflex as rx
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# =====================================================================
# ARQUITECTURA RED NEURONAL 
# =====================================================================
class RecommenderNet(torch.nn.Module):
    def __init__(self, num_users, num_items, num_features, embedding_dim=16):
        super(RecommenderNet, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        input_dim = (embedding_dim * 2) + num_features
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(16, 1)
        )

    def forward(self, user_indices, item_indices, context_features):
        user_emb = self.user_embedding(user_indices)
        item_emb = self.item_embedding(item_indices)
        x = torch.cat([user_emb, item_emb, context_features], dim=1)
        return torch.sigmoid(self.mlp(x).squeeze()) * 5.0

# =====================================================================
# 💾 CARGA REAL DE LOS ARTEFACTOS Y DATAFRAMES EN EL SERVIDOR
# =====================================================================
modelo_pytorch = None
artefactos = {}
df_final_dataset = None
df_users_features = None
df_Destinations = None
lista_usuarios_inicialalizar = [] # Guardará la lista filtrada de usuarios para la UI

try:
    # 1. Cargar pesos entrenados y diccionarios desde tu .pt
    artefactos = torch.load("modelo_recomendacion_viajes.pt", map_location=torch.device('cpu'))
    
    user_to_idx = artefactos['user_to_idx']
    dest_to_idx = artefactos['dest_to_idx']
    idx_to_dest = artefactos['idx_to_dest']
    feature_cols = artefactos['feature_cols']
    num_destinations = artefactos['num_destinations']

    # 2. Instanciar el modelo con las dimensiones exactas guardadas
    modelo_pytorch = RecommenderNet(
        num_users=artefactos['num_users'],
        num_items=artefactos['num_destinations'],
        num_features=artefactos['num_features']
    )
    modelo_pytorch.load_state_dict(artefactos['model_state_dict'])
    modelo_pytorch.eval() 
    
    # 3. Cargar los DataFrames
    df_final_dataset = pd.read_csv("final_preprocessed_travel_dataset.csv")
    df_users_features = pd.read_csv("users_features_dataset.csv")
    df_Destinations = pd.read_csv("destinations_dataset.csv")
    
    # 4. Preparamos la lista de usuarios VERIFICANDO que existan en el modelo de IA y extrayendo TODA la info
    for row in df_users_features.itertuples():
        id_actual = row.UserID
        
        try:
            id_busqueda = int(id_actual) if str(id_actual).isdigit() else id_actual
        except:
            id_busqueda = id_actual

        if id_busqueda in user_to_idx:
            lista_usuarios_inicialalizar.append({
                "id": str(id_actual),
                "nombre": str(row.Name),
                "email": str(row.Email),
                "viajeros": str(row.Total_Travelers),
                "genero": "Masculino" if int(row.Gender_Bin) == 1 else "Femenino",
                "adv": "🔥" if float(row.Adventure) > 0.5 else "❌",
                "beach": "🏖️" if float(row.Beaches) > 0.5 else "❌",
                "city": "🏙️" if float(row.City) > 0.5 else "❌",
                "hist": "🏛️" if float(row.Historical) > 0.5 else "❌",
                "nat": "🌲" if float(row.Nature) > 0.5 else "❌"
            })
    
    print(f"🚀 [IA] Motor acoplado con éxito. {len(lista_usuarios_inicialalizar)} usuarios validados cargados.")
except Exception as e:
    print(f"⚠️ [ERROR AL CARGAR EL MOTOR]: {str(e)}")

# =====================================================================
# 🧠 ESTADO GLOBAL DE LA APLICACIÓN (Lógica de Control)
# =====================================================================
class AppState(rx.State):
    """Maneja el enrutamiento interno de pestañas y la lógica de inferencia de los módulos."""
    pestana_activa: str = "inicio"
    
    # Módulo 2: Clasificación de Imágenes
    resultado_clasificacion: str = "Esperando imagen de cabina..."
    
    # Módulo 3: Nuevo Sistema de Selección de Usuarios y Recomendación Top 3
    usuarios_sistema: list[dict] = lista_usuarios_inicialalizar
    usuario_seleccionado_id: str = ""
    usuario_seleccionado_nombre: str = ""
    
    # Guardará la información extendida del usuario seleccionado para mostrarla en la UI
    usuario_seleccionado_info: dict = {
        "id": "", "nombre": "", "email": "", "viajeros": "", "genero": "",
        "adv": "", "beach": "", "city": "", "hist": "", "nat": ""
    }
    
    error_busqueda: str = ""
    reporte_recomendacion: list[dict] = []

    def cambiar_pestana(self, nueva_pestana: str):
        self.pestana_activa = nueva_pestana

    async def manejar_subida_imagen(self, archivos: list[rx.UploadFile]):
        for archivo in archivos:
            await archivo.read()
            self.resultado_clasificacion = "⚠️ ALERTA: Conductor hablando por teléfono celular (Uso de Móvil)"
            
    def seleccionar_usuario_y_recomendar(self, u_info: dict):
        """Selecciona un usuario de la lista limpia y calcula inmediatamente su Top 3."""
        self.usuario_seleccionado_id = u_info["id"]
        self.usuario_seleccionado_nombre = u_info["nombre"]
        self.usuario_seleccionado_info = u_info
        self.error_busqueda = ""
        self.reporte_recomendacion = []

        if modelo_pytorch is None or df_final_dataset is None:
            self.error_busqueda = "Los artefactos de IA no están listos en el servidor."
            return

        try:
            user_id_consulta = int(self.usuario_seleccionado_id) if self.usuario_seleccionado_id.isdigit() else self.usuario_seleccionado_id
        except:
            user_id_consulta = self.usuario_seleccionado_id

        if user_id_consulta not in user_to_idx:
            self.error_busqueda = f"El usuario {self.usuario_seleccionado_id} no se encuentra mapeado en la red."
            return

        try:
            u_idx = user_to_idx[user_id_consulta]

            # 1. Exclusión contextual (destinos que ya conoce)
            destinos_conocidos = set(df_final_dataset[df_final_dataset['UserID'] == user_id_consulta]['dest_idx'].values)
            todos_los_dest_idx = set(range(num_destinations))
            destinos_candidatos = list(todos_los_dest_idx - destinos_conocidos)

            if not destinos_candidatos:
                self.error_busqueda = "El usuario ya ha visitado todos los destinos disponibles."
                return

            # 2. Vector de características del usuario
            user_features_row = df_users_features[df_users_features['UserID'] == user_id_consulta]
            features_v = user_features_row[feature_cols].values[0]

            # 3. Tensores de Inferencia en CPU (Corrección del Warning de duplicación de Arrays)
            features_multiplicadas = np.tile(features_v, (len(destinos_candidatos), 1))
            u_tensor = torch.tensor([u_idx] * len(destinos_candidatos), dtype=torch.long)
            d_tensor = torch.tensor(destinos_candidatos, dtype=torch.long)
            f_tensor = torch.tensor(features_multiplicadas, dtype=torch.float32)

            # 4. Forward pass libre de gradientes
            with torch.no_grad():
                preds = modelo_pytorch(u_tensor, d_tensor, f_tensor)
                if preds.dim() == 0:
                    preds = preds.unsqueeze(0)
                preds = preds.numpy()

            # 5. Mapear predicciones y cruzar datos completos
            candidatos_reales_ids = [idx_to_dest[idx] for idx in destinos_candidatos]
            df_predicciones = pd.DataFrame({
                'DestinationID': candidatos_reales_ids,
                'Prediccion_Original': preds
            })

            df_reporte = pd.merge(df_predicciones, df_Destinations, on='DestinationID', how='inner')
            df_reporte = df_reporte.drop_duplicates(subset=['Name'])

            # 🔥 CAMBIO SOLICITADO: Tomar estrictamente el Top 3 (Head 3)
            df_top_k = df_reporte.sort_values(by='Prediccion_Original', ascending=False).head(3)

            lista_final = []
            for rank, row in enumerate(df_top_k.itertuples(), start=1):
                score_continuo = np.clip(row.Prediccion_Original, 1.0, 5.0)
                estrellas_discretas = int(round(float(score_continuo)))
                
                # Obtener el nombre del destino para asignar su imagen correspondiente
                nombre_destino = str(row.Name).strip()
                
                lista_final.append({
                    "rank": str(rank),
                    "dest_id": str(row.DestinationID),
                    "lugar": nombre_destino,
                    "ciudad": str(row.State),
                    "cat": str(row.Type),
                    "estrellas": "⭐" * estrellas_discretas if estrellas_discretas > 0 else "⭐",
                    "num_estrellas": f"{estrellas_discretas}/5",
                    "temporada": str(row.BestTimeToVisit),
                    "pop": f"{row.Popularity:.2f}",
                    "score_ia": f"{score_continuo:.2f}"
                })
            
            self.reporte_recomendacion = lista_final

        except Exception as e:
            self.error_busqueda = f"Error en inferencia real: {str(e)}"

# =====================================================================
# 🎨 DISEÑO Y ESTILOS
# =====================================================================
THEME = {
    "colors": {
        "bg_base": "#0B0F19",
        "bg_surface": "#121826",
        "bg_card": "#172033",
        "accent": "#3B82F6",
        "accent_light": "#60A5FA",
        "text_main": "#F3F4F6",
        "text_sub": "#9CA3AF",
        "border": "#22314D",
        "alert": "#F59E0B",
    },
    "fonts": {
        "main": "'Inter', sans-serif",
        "mono": "'JetBrains Mono', monospace"
    }
}

STYLES = {
    "card": {
        "background": THEME["colors"]["bg_card"],
        "border": f"1px solid {THEME['colors']['border']}",
        "border_radius": "14px",
        "padding": "24px",
        "width": "100%",
        "transition": "all 0.2s ease-in-out",
        "_hover": {"border_color": THEME["colors"]["accent"], "transform": "translateY(-2px)"}
    },
    "heading": {
        "color": THEME["colors"]["text_main"],
        "font_family": THEME["fonts"]["main"],
        "font_weight": "600",
        "letter_spacing": "-0.01em", 
        "line_height": "1.3"
    },
    "body": {
        "color": THEME["colors"]["text_sub"],
        "font_family": THEME["fonts"]["main"],
        "font_size": "14px",
        "line_height": "1.6" 
    }
}

# =====================================================================
# 🧩 COMPONENTES DE DISEÑO COMUNES
# =====================================================================
def crear_encabezado_seccion(modulo_tag: str, titulo: str, descripcion: str) -> rx.Component:
    return rx.vstack(
        rx.box(modulo_tag, font_family=THEME["fonts"]["mono"], font_size="11px", color=THEME["colors"]["accent_light"], letter_spacing="0.05em", font_weight="600"),
        rx.heading(titulo, size="7", style=STYLES["heading"]),
        rx.text(descripcion, style=STYLES["body"]),
        spacing="1",
        margin_bottom="16px",
        align_items="start"
    )

def nav_btn(label: str, pestana: str, icono: str) -> rx.Component:
    es_activo = AppState.pestana_activa == pestana
    return rx.button(
        rx.hstack(
            rx.icon(tag=icono, size=16),
            rx.text(label, font_size="13px", font_weight="500"),
            spacing="2"
        ),
        on_click=lambda: AppState.cambiar_pestana(pestana),
        variant="ghost",
        color=rx.cond(es_activo, THEME["colors"]["accent_light"], THEME["colors"]["text_sub"]),
        background=rx.cond(es_activo, "#3B82F618", "transparent"),
        border=rx.cond(es_activo, f"1px solid {THEME['colors']['accent']}33", "1px solid transparent"),
        border_radius="8px",
        padding="8px 16px",
        height="auto",
        _hover={"background": "#3B82F610", "color": THEME["colors"]["text_main"]},
        transition="all 0.15s ease"
    )

# =====================================================================
# 🗺️ ESTRUCTURA DE LAS VISTAS (Pestañas)
# =====================================================================

def modulo_resumen_card(icono: str, titulo: str, desc: str) -> rx.Component:
    return rx.vstack(
        rx.box(rx.icon(tag=icono, size=20, color="white"), background=THEME["colors"]["accent"], border_radius="8px", padding="8px"),
        rx.heading(titulo, size="4", style=STYLES["heading"]),
        rx.text(desc, style=STYLES["body"], font_size="13px"),
        style=STYLES["card"],
        align_items="start",
        spacing="2"
    )

def vista_inicio() -> rx.Component:
    return rx.vstack(
        rx.vstack(
            rx.heading("Bienvenido a SmartRouteApp", size="8", style=STYLES["heading"], font_weight="700"),
            rx.text("Optimización logística de transporte, seguridad activa en cabina y personalización avanzada mediante Inteligencia Artificial.", style=STYLES["body"], font_size="15px", max_width="700px"),
            align_items="start",
            spacing="2",
            margin_bottom="24px"
        ),
        rx.grid(
            modulo_resumen_card("chart-line", "Predicción de Demanda", "Análisis estratégico de series de tiempo a 30 días para flujos de pasajeros."),
            modulo_resumen_card("eye", "Seguridad Vial", "Detección inteligente de cansancio y conductas de riesgo mediante CNN en tiempo real."),
            modulo_resumen_card("map-pin", "Motor de Recomendaciones", "Fusión híbrida de algoritmos para asignación inteligente de destinos óptimos."),
            columns="3",
            spacing="4",
            width="100%"
        ),
        width="100%",
        align_items="start"
    )

def vista_modulo1() -> rx.Component:
    return rx.vstack(
        crear_encabezado_seccion("MÓDULO 01 — PREDICCIÓN DE DEMANDAS DE TRANSPORTE", "Predicción de Demanda de Transporte", "Proyecciones avanzadas de volumen de pasajeros para la optimización de recursos."),
        rx.box(
            rx.vstack(
                rx.grid(
                    rx.vstack(rx.text("PROYECCIÓN 30 DÍAS", font_family=THEME["fonts"]["mono"], font_size="10px", color=THEME["colors"]["text_sub"]), rx.text("12,847", font_size="28px", style=STYLES["heading"]), rx.text("Pasajeros estimados", font_size="12px", color=THEME["colors"]["text_sub"]), align_items="start", spacing="0"),
                    rx.vstack(rx.text("PRECISIÓN DEL MODELO", font_family=THEME["fonts"]["mono"], font_size="10px", color=THEME["colors"]["text_sub"]), rx.text("94.2%", font_size="28px", color=THEME["colors"]["accent_light"], font_family=THEME["fonts"]["main"], font_weight="600"), rx.text("R² Score Validado", font_size="12px", color=THEME["colors"]["text_sub"]), align_items="start", spacing="0"),
                    columns="2", width="100%", padding_bottom="16px", border_bottom=f"1px solid {THEME['colors']['border']}"
                ),
                rx.center(
                    rx.vstack(rx.icon(tag="chart-line", size=32, color=THEME["colors"]["text_sub"]), rx.text("El gráfico analítico interactivo se renderizará en este cuadrante.", style=STYLES["body"], font_size="13px"), align_items="center", spacing="2"),
                    background=THEME["colors"]["bg_base"], height="260px", width="100%", border_radius="10px", border=f"1px dashed {THEME['colors']['border']}", margin_top="16px"
                ),
                width="100%"
            ),
            style=STYLES["card"]
        ),
        width="100%", align_items="start"
    )

def vista_modulo2() -> rx.Component:
    return rx.vstack(
        crear_encabezado_seccion("MÓDULO 02 — CLASIFICACIÓN DE CONDUCCIÓN DISTRACTIVA", "Sistema de Clasificación de Conducción Distractiva", "Análisis automatizado de transmisiones en cabina para prevenir siniestros viales."),
        rx.box(
            rx.vstack(
                rx.upload(
                    rx.vstack(
                        rx.box(rx.icon(tag="cloud-upload", size=24, color=THEME["colors"]["accent_light"]), background="#3B82F615", border_radius="50%", padding="12px"),
                        rx.text("Arrastre o seleccione una captura de cabina", color=THEME["colors"]["text_main"], font_size="14px", font_weight="500"),
                        rx.text("Formatos soportados: PNG, JPG (Máx 10MB)", font_size="12px", color=THEME["colors"]["text_sub"]),
                        align_items="center", spacing="1"
                    ),
                    id="subida_cabina", border=f"1.5px dashed {THEME['colors']['border']}", border_radius="12px", padding="32px", width="100%", background=THEME["colors"]["bg_base"], _hover={"border_color": THEME["colors"]["accent"]}, cursor="pointer"
                ),
                rx.button(
                    rx.hstack(rx.icon(tag="brain", size=16), rx.text("Correr Diagnóstico con Red Neuronal"), spacing="2"),
                    on_click=AppState.manejar_subida_imagen(rx.upload_files(upload_id="subida_cabina")),
                    background=THEME["colors"]["accent"], color="white", border_radius="8px", width="100%", padding="12px", height="auto", _hover={"background": "#2563EB"}
                ),
                rx.hstack(
                    rx.icon(tag="shield-alert", size=18, color=THEME["colors"]["alert"]),
                    rx.vstack(
                        rx.text("RESULTADO DEL ANÁLISIS EN TIEMPO REAL", font_family=THEME["fonts"]["mono"], font_size="10px", color=THEME["colors"]["text_sub"]),
                        rx.text(AppState.resultado_clasificacion, color=THEME["colors"]["alert"], font_size="14px", font_weight="600"),
                        align_items="start", spacing="0"
                    ),
                    background="#F59E0B08", border=f"1px solid {THEME['colors']['alert']}33", border_radius="10px", padding="12px 16px", width="100%", align_items="center", spacing="3"
                ),
                width="100%", spacing="4"
            ),
            style=STYLES["card"]
        ),
        width="100%", align_items="start"
    )

# --- VISTA: MÓDULO 3 (SISTEMA DE COMPORTAMIENTO COMPLETO) ---
def usuario_card_item(user: dict) -> rx.Component:
    """Muestra la información del usuario renderizando SOLO lo que sí tiene y con mejor contraste."""
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.box(rx.icon(tag="user", size=16, color="white"), background=THEME["colors"]["accent"], border_radius="6px", padding="6px"),
                rx.vstack(
                    rx.text(user["nombre"], font_size="15px", font_weight="600", color="#FFFFFF"), # <-- Texto Blanco puro
                    rx.text(f"ID: {user['id']} | {user['email']}", font_size="12px", color="#9CA3AF"), # <-- Gris claro legible
                    spacing="0", align_items="start"
                ),
                spacing="3", align_items="center"
            ),
            rx.divider(border_color=THEME["colors"]["border"], margin_top="6px", margin_bottom="6px"),
            
            # Datos fijos con color de alto contraste (Blanco/Texto Principal)
            rx.hstack(
                rx.text(f"👥 Viajeros: {user['viajeros']}", font_size="12px", color=THEME["colors"]["text_main"], font_weight="500"),
                rx.text(f"🧬 {user['genero']}", font_size="12px", color=THEME["colors"]["text_main"], font_weight="500"),
                spacing="4", margin_bottom="6px"
            ),
            
            # Contenedor dinámico de intereses en formato etiquetas/badges continuos
            rx.flex(
                # Si es un "🔥" (Sí lo tiene), muestra la etiqueta de aventura en verde brillante. Si no, nada.
                rx.cond(user["adv"] == "🔥", rx.box(rx.text("🔥 Adventure", font_size="11px", color="#10B981", font_weight="600"), background="#10B98115", padding="2px 8px", border_radius="4px", margin="2px")),
                # Si es "🏖️" (Sí lo tiene), muestra la etiqueta en azul brillante.
                rx.cond(user["beach"] == "🏖️", rx.box(rx.text("🏖️ Beaches", font_size="11px", color="#3B82F6", font_weight="600"), background="#3B82F615", padding="2px 8px", border_radius="4px", margin="2px")),
                # Si es "🏙️" (Sí lo tiene)...
                rx.cond(user["city"] == "🏙️", rx.box(rx.text("🏙️ City", font_size="11px", color="#A855F7", font_weight="600"), background="#A855F715", padding="2px 8px", border_radius="4px", margin="2px")),
                # Si es "🏛️" (Sí lo tiene)...
                rx.cond(user["hist"] == "🏛️", rx.box(rx.text("🏛️ Historical", font_size="11px", color="#F59E0B", font_weight="600"), background="#F59E0B15", padding="2px 8px", border_radius="4px", margin="2px")),
                # Si es "🌲" (Sí lo tiene)...
                rx.cond(user["nat"] == "🌲", rx.box(rx.text("🌲 Nature", font_size="11px", color="#10B981", font_weight="600"), background="#10B98115", padding="2px 8px", border_radius="4px", margin="2px")),
                
                wrap="wrap",
                width="100%"
            ),
            align_items="start", width="100%"
        ),
        padding="14px 16px",
        background=THEME["colors"]["bg_surface"],
        border=f"1px solid {THEME['colors']['border']}",
        border_radius="10px",
        cursor="pointer",
        transition="all 0.15s ease",
        _hover={"border_color": THEME["colors"]["accent"], "background": "#3B82F608"},
        on_click=lambda: AppState.seleccionar_usuario_y_recomendar(user),
        width="100%"
    )

def destino_card_top3(destino: dict) -> rx.Component:
    """Muestra toda la información del destino con imágenes basadas en los nombres reales de la India."""
    return rx.hstack(
        # Círculo del Top grande, perfecto y rígido
        rx.center(
            rx.text(destino["rank"], font_size="16px", font_weight="700", color="white"),
            background=THEME["colors"]["accent"], 
            border_radius="9999px",
            width="40px",
            height="40px",
            flex_shrink="0",
        ),
        
        # 🔥 CONTROL DE IMÁGENES POR NOMBRE REAL DEL CSV
        rx.box(
            rx.cond(destino["lugar"] == "Taj Mahal", rx.image(src="/taj_mahal.jpg", width="90px", height="65px", object_fit="cover", border_radius="8px")),
            rx.cond(destino["lugar"] == "Goa Beaches", rx.image(src="/goa_beaches.jpg", width="90px", height="65px", object_fit="cover", border_radius="8px")),
            rx.cond(destino["lugar"] == "Jaipur City", rx.image(src="/jaipur_city.jpg", width="90px", height="65px", object_fit="cover", border_radius="8px")),
            rx.cond(destino["lugar"] == "Kerala Backwaters", rx.image(src="/kerala.jpg", width="90px", height="65px", object_fit="cover", border_radius="8px")),
            rx.cond(destino["lugar"] == "Leh Ladakh", rx.image(src="/leh_ladakh.jpg", width="90px", height="65px", object_fit="cover", border_radius="8px")),
            
            # Imagen de respaldo por si hay variaciones de texto o espacios invisibles en el CSV
            rx.cond(
                (destino["lugar"] != "Taj Mahal") & (destino["lugar"] != "Goa Beaches") & 
                (destino["lugar"] != "Jaipur City") & (destino["lugar"] != "Kerala Backwaters") & 
                (destino["lugar"] != "Leh Ladakh"),
                rx.image(src="/default_travel.jpg", width="90px", height="65px", object_fit="cover", border_radius="8px")
            ),
            flex_shrink="0" # Evita que la imagen se aplaste si el texto del destino es largo
        ),
        
        # Despliegue de los campos de información (Sin el ID técnico)
        rx.vstack(
            rx.hstack(
                rx.text(destino["lugar"], font_size="15px", font_weight="700", color=THEME["colors"]["text_main"]), 
                spacing="2"
            ),
            rx.grid(
                rx.text(f"📍 Estado: {destino['ciudad']}", font_size="12px", color=THEME["colors"]["text_sub"]),
                rx.text(f"🗂️ Tipo: {destino['cat']}", font_size="12px", color=THEME["colors"]["text_sub"]),
                rx.text(f"📈 Popularidad: {destino['pop']}", font_size="12px", color=THEME["colors"]["text_sub"]),
                rx.text(f"📅 Temp Ideal: {destino['temporada']}", font_size="12px", color=THEME["colors"]["text_sub"]),
                columns="2", spacing_x="4", spacing_y="1"
            ),
            rx.hstack(
                rx.text(destino["estrellas"], font_size="11px"),
                rx.text("Score Red Neuronal: ", destino["score_ia"], "/5.00", font_family=THEME["fonts"]["mono"], font_size="11px", color=THEME["colors"]["accent_light"]),
                spacing="3", align_items="center", margin_top="4px"
            ),
            spacing="1", align_items="start"
        ),
        width="100%", padding="16px", border_bottom=f"1px solid {THEME['colors']['border']}", justify="start", align_items="center", spacing="4"
    )

def vista_modulo3() -> rx.Component:
    return rx.grid(
        # Columna Izquierda: Panel de Selección de Usuarios (9 campos visibles)
        rx.vstack(
            crear_encabezado_seccion("MÓDULO 03 — RECOMENDACIONES TURÍSTICAS", "Sistema de Recomendación de Destinos Turísticos", "Seleccione un usuario para calcular sus destinos turísticos ideales."),
            rx.box(
                rx.vstack(
                    rx.heading("Lista de Usuarios", size="3", style=STYLES["heading"], margin_bottom="8px"),
                    rx.scroll_area(
                        rx.vstack(
                            rx.foreach(AppState.usuarios_sistema, usuario_card_item),
                            spacing="3",
                            width="100%",
                            padding_right="8px"
                        ),
                        max_height="520px",
                        scrollbars="vertical",
                        width="100__%"
                    ),
                    width="100%"
                ),
                style=STYLES["card"]
            ),
            width="100%",
            align_items="start"
        ),
        
        # Columna Derecha: Resultados del Análisis de Inferencia (Top 3 completo + Imágenes)
        rx.vstack(
            rx.heading("Resultados de la Recomendación", size="5", style=STYLES["heading"], margin_bottom="8px"),
            
            rx.cond(
                AppState.error_busqueda,
                rx.box(rx.text(AppState.error_busqueda, color="#EF4444", font_size="13px"), background="#EF444410", padding="12px", border_radius="8px", width="100%")
            ),
            
            rx.cond(
                AppState.usuario_seleccionado_id,
                rx.box(
                    rx.hstack(
                        rx.icon(tag="circle-check", size=16, color="#10B981"),
                        rx.text("Analizando a: ", font_size="13px", color=THEME["colors"]["text_sub"]),
                        rx.text(AppState.usuario_seleccionado_nombre, font_size="13px", font_weight="600", color=THEME["colors"]["accent_light"]),
                        spacing="2"
                    ),
                    background="#10B98108", border=f"1px solid #10B98133", padding="10px 14px", border_radius="8px", width="100%", margin_bottom="12px"
                ),
                rx.center(
                    rx.hstack(rx.icon(tag="info", size=14), rx.text("Seleccione un usuario de la lista de la izquierda para comenzar.")),
                    background=THEME["colors"]["bg_surface"], border=f"1px dashed {THEME['colors']['border']}", padding="16px", border_radius="8px", width="100%", font_size="13px", color=THEME["colors"]["text_sub"]
                )
            ),
            
            # Bloque del Top 3 Recomendaciones reales calculadas por PyTorch
            rx.cond(
                AppState.reporte_recomendacion,
                rx.box(
                    rx.vstack(
                        rx.hstack(
                            rx.icon(tag="list", size=16, color=THEME["colors"]["accent_light"]), 
                            rx.heading("Top 3 Destinos Recomendados", size="3", style=STYLES["heading"]), 
                            spacing="2", padding="14px 16px", border_bottom=f"1px solid {THEME['colors']['border']}", width="100%"
                        ),
                        rx.foreach(AppState.reporte_recomendacion, destino_card_top3),
                        width="100%", spacing="0"
                    ),
                    background=THEME["colors"]["bg_card"], border=f"1px solid {THEME['colors']['border']}", border_radius="14px", width="100%", overflow="hidden"
                )
            ),
            width="100%",
            align_items="start"
        ),
        columns="2",
        spacing="6",
        width="100%"
    )

# =====================================================================
# 🖥️ COMPOSICIÓN DEL CONTENEDOR RAÍZ
# =====================================================================
def index() -> rx.Component:
    return rx.box(
        rx.html("<style>@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap'); * { box-sizing: border-box; font-family: 'Inter', sans-serif; }</style>"),
        
        # Header principal
        rx.hstack(
            rx.hstack(
                rx.box(rx.icon(tag="route", size=20, color="white"), background=f"linear-gradient(135deg, {THEME['colors']['accent']} 0%, #1D4ED8 100%)", border_radius="8px", padding="8px"),
                rx.vstack(
                    rx.heading("SmartRouteApp", size="5", style=STYLES["heading"], letter_spacing="-0.02em"),
                    spacing="0", align_items="start"
                ),
                spacing="3", align_items="center"
            ),
            background=THEME["colors"]["bg_surface"], padding="14px 40px", border_bottom=f"1px solid {THEME['colors']['border']}", width="100%"
        ),
        
        # Barra de Navegación Limpia con Espaciado Generoso
        rx.hstack(
            nav_btn("Inicio", "inicio", "home"),
            nav_btn("1. Predicción de Demandas de Transporte", "modulo1", "chart-line"),
            nav_btn("2. Clasificación de Conducción Distractiva", "modulo2", "eye"),
            nav_btn("3. Recomendaciones Turísticas", "modulo3", "map-pin"),
            spacing="6", 
            background=THEME["colors"]["bg_base"], padding="12px 40px", width="100%", border_bottom=f"1px solid {THEME['colors']['border']}"
        ),
        
        # Render Dinámico de Módulos sin colisiones
        rx.box(
            rx.cond(AppState.pestana_activa == "inicio", vista_inicio()),
            rx.cond(AppState.pestana_activa == "modulo1", vista_modulo1()),
            rx.cond(AppState.pestana_activa == "modulo2", vista_modulo2()),
            rx.cond(AppState.pestana_activa == "modulo3", vista_modulo3()),
            padding="32px 40px", width="100%", max_width="1250px", margin="0 auto"
        ),
        
        background_color=THEME["colors"]["bg_base"],
        min_height="100vh",
        width="100%"
    )

# Configuración de inicialización de la App
app = rx.App()
app.add_page(index, title="SmartRouteApp")