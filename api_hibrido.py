from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
import joblib
import os
import uvicorn

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
MODELS_DIR = "models/"
cerebros = {}

# =============================================================================
# GESTOR DE CICLO DE VIDA (Lifespan)
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n🧠 INICIANDO MOTOR DE IA - HÍBRIDO (FOLLOWERS + VIEWERS)...")
    print(f"📂 Buscando modelos en: {MODELS_DIR}")
    
    # --- CARGAR TWITCH ---
    try:
        path_fol = os.path.join(MODELS_DIR, 'twitch_modelo_followers.pkl')
        path_fol_scaler = os.path.join(MODELS_DIR, 'twitch_scaler_followers.pkl')
        path_view = os.path.join(MODELS_DIR, 'twitch_modelo_viewers.pkl')
        path_view_scaler = os.path.join(MODELS_DIR, 'twitch_scaler_viewers.pkl')
        
        if os.path.exists(path_fol) and os.path.exists(path_fol_scaler):
            cerebros['t_fol_model'] = joblib.load(path_fol)
            cerebros['t_fol_scaler'] = joblib.load(path_fol_scaler)
            print("✅ Twitch Followers: Modelo y escalador cargados.")
        else:
            print(f"⚠️ Twitch Followers: Archivos no encontrados")
            
        if os.path.exists(path_view) and os.path.exists(path_view_scaler):
            cerebros['t_view_model'] = joblib.load(path_view)
            cerebros['t_view_scaler'] = joblib.load(path_view_scaler)
            print("✅ Twitch Viewers: Modelo y escalador cargados.")
        else:
            print(f"⚠️ Twitch Viewers: Archivos no encontrados")
            
    except Exception as e:
        print(f"❌ Twitch: Error al cargar modelos: {e}")

    # --- CARGAR KICK ---
    try:
        path_fol_k = os.path.join(MODELS_DIR, 'kick_modelo_followers.pkl')
        path_fol_k_scaler = os.path.join(MODELS_DIR, 'kick_scaler_followers.pkl')
        path_view_k = os.path.join(MODELS_DIR, 'kick_modelo_viewers.pkl')
        path_view_k_scaler = os.path.join(MODELS_DIR, 'kick_scaler_viewers.pkl')
        
        if os.path.exists(path_fol_k) and os.path.exists(path_fol_k_scaler):
            cerebros['k_fol_model'] = joblib.load(path_fol_k)
            cerebros['k_fol_scaler'] = joblib.load(path_fol_k_scaler)
            print("✅ Kick Followers: Modelo y escalador cargados.")
        else:
            print(f"⚠️ Kick Followers: Archivos no encontrados")
            
        if os.path.exists(path_view_k) and os.path.exists(path_view_k_scaler):
            cerebros['k_view_model'] = joblib.load(path_view_k)
            cerebros['k_view_scaler'] = joblib.load(path_view_k_scaler)
            print("✅ Kick Viewers: Modelo y escalador cargados.")
        else:
            print(f"⚠️ Kick Viewers: Archivos no encontrados")
            
    except Exception as e:
        print(f"❌ Kick: Error al cargar modelos: {e}")

    yield
    cerebros.clear()
    print("🛑 Motor de IA apagado.")

app = FastAPI(
    title="API Híbrida de Predicción (Followers + Viewers)",
    description="Combina lógica de viewers (api_ia_twitch.py) con followers (api_ia_twitch_v2.py)",
    version="1.0",
    lifespan=lifespan
)

# =============================================================================
# DTOs
# =============================================================================
class TwitchInput(BaseModel):
    """
    Input para Twitch con 8 features.
    - Followers: predicción con LOG
    - Viewers: predicción con escaladores (sin LOG)
    """
    avgViewers_d1: float
    avgViewers_d14: float
    minutesStreamed_d1: float
    minutesStreamed_d14: float
    followers_d1: float
    followers_d14: float
    comments_most_viewed: float
    comments_least_viewed: float

class KickInput(BaseModel):
    """
    Input para Kick con 6 features.
    """
    AVG_VIEWERS_D1: float
    AVG_VIEWERS_D14: float
    HOURS_STREAMED_D1: float
    HOURS_STREAMED_D14: float
    FOLLOWERS_D14: float
    FOLLOWERS_D21: float

# =============================================================================
# ENDPOINTS
# =============================================================================
@app.get("/")
def home():
    return {
        "mensaje": "Motor de IA Híbrido activo",
        "endpoints": {
            "/predecir/twitch": "POST - Followers (LOG) + Viewers (Escalador)",
            "/predecir/kick": "POST - Followers (LOG)",
            "/health": "GET - Estado de modelos",
            "/docs": "Documentación interactiva"
        }
    }

@app.post("/predecir/twitch")
def predict_twitch(data: TwitchInput):
    """
    Predicción HÍBRIDA para Twitch:
    
    FOLLOWERS: 
    - Usa LOGARITMO (lógica api_ia_twitch_v2.py)
    - followers_d30 = followers_d1 + crecimiento_log
    
    VIEWERS:
    - Usa ESCALADORES directos (lógica api_ia_twitch.py)
    - avg_viewers_d30 predicción directa escalada
    """
    
    # Validar que al menos Viewers esté cargado
    if 't_view_model' not in cerebros or 't_view_scaler' not in cerebros:
        raise HTTPException(
            status_code=503,
            detail="Modelo Viewers de Twitch no cargado"
        )

    try:
        input_dict = data.dict()
        df = pd.DataFrame([input_dict])
        
        # =====================================================================
        # PARTE 1: PREDICCIÓN DE VIEWERS (lógica api_ia_twitch.py)
        # =====================================================================
        if 't_view_model' in cerebros and 't_view_scaler' in cerebros:
            X_scaled_view = cerebros['t_view_scaler'].transform(df)
            pred_viewers = cerebros['t_view_model'].predict(X_scaled_view)[0]
            
            # Validar finitud
            if not np.isfinite(pred_viewers):
                pred_viewers = 0
        else:
            pred_viewers = 0
            print("⚠️ Modelo Viewers no disponible")
        
        # =====================================================================
        # PARTE 2: PREDICCIÓN DE FOLLOWERS (lógica api_ia_twitch_v2.py con LOG)
        # =====================================================================
        if 't_fol_model' in cerebros:
            # Aplicar LOG a todas las columnas
            df_log = df.copy()
            for col in df_log.columns:
                df_log[f'{col}_log'] = np.log1p(df_log[col].astype(float))
            
            # Seleccionar columnas logarítmicas
            cols_log = [
                'avgViewers_d1_log', 'avgViewers_d14_log',
                'minutesStreamed_d1_log', 'minutesStreamed_d14_log',
                'followers_d1_log', 'followers_d14_log',
                'comments_most_viewed_log', 'comments_least_viewed_log'
            ]
            cols_disponibles = [c for c in cols_log if c in df_log.columns]
            X_input_fol = df_log[cols_disponibles]
            
            # Predicción
            pred_crecimiento_log = cerebros['t_fol_model'].predict(X_input_fol)[0]
            
            # Limitar logaritmo
            pred_crecimiento_log = np.clip(pred_crecimiento_log, None, 20)
            
            # Convertir de LOG a Real
            crecimiento_real = np.expm1(pred_crecimiento_log)
            if not np.isfinite(crecimiento_real):
                crecimiento_real = 0
            
            followers_d30 = data.followers_d1 + crecimiento_real
            
            # =====================================================================
            # SANITY CHECK INTELIGENTE: Validar crecimiento razonable
            # =====================================================================
            # Calcular porcentaje de crecimiento en relación al base actual
            if data.followers_d1 > 0:
                pct_crecimiento = (crecimiento_real / data.followers_d1) * 100
                
                # Limites según tamaño del canal
                if data.followers_d1 >= 1_000_000:
                    # Canales gigantes: máx 1% de crecimiento
                    max_pct = 1.0
                elif data.followers_d1 >= 500_000:
                    # Canales grandes: máx 2% de crecimiento
                    max_pct = 2.0
                elif data.followers_d1 >= 100_000:
                    # Canales medianos: máx 5% de crecimiento
                    max_pct = 5.0
                else:
                    # Canales pequeños: máx 10% de crecimiento
                    max_pct = 10.0
                
                # Si excede el límite, ajustar
                if pct_crecimiento > max_pct:
                    print(f"⚠️ Crecimiento anómalo detectado: {pct_crecimiento:.2f}%. Limitando a {max_pct}%")
                    crecimiento_real = (data.followers_d1 * max_pct) / 100
                    followers_d30 = data.followers_d1 + crecimiento_real
                
        else:
            followers_d30 = data.followers_d1
            crecimiento_real = 0
            print("⚠️ Modelo Followers no disponible")
        
        return {
            "status": "success",
            "plataforma": "Twitch",
            "prediccion": {
                "followers_d30": int(followers_d30),
                "avg_viewers_d30": int(pred_viewers),
                "debug": {
                    "followers_d1": int(data.followers_d1),
                    "crecimiento_neto": int(crecimiento_real),
                    "crecimiento_porcentaje": round((crecimiento_real / data.followers_d1 * 100), 2) if data.followers_d1 > 0 else 0
                }
            },
            "input_received": input_dict
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=400,
            detail=f"Error en predicción Twitch: {str(e)}"
        )

@app.post("/predecir/kick")
def predict_kick(data: KickInput):
    """
    Predicción HÍBRIDA para Kick:
    
    FOLLOWERS: 
    - Usa LOGARITMO (lógica api_ia_twitch_v2.py)
    - followers_d28 = followers_d14 + crecimiento_log
    
    VIEWERS:
    - Usa ESCALADORES directos (lógica api_ia_twitch.py)
    - avg_viewers_d30 predicción directa escalada
    
    Input: 6 features (sin comentarios)
    """
    
    if 'k_fol_model' not in cerebros:
        raise HTTPException(
            status_code=503,
            detail="Modelo Followers de Kick no cargado"
        )

    try:
        input_dict = data.dict()
        df = pd.DataFrame([input_dict])
        
        # =====================================================================
        # PARTE 1: PREDICCIÓN DE VIEWERS (lógica api_ia_twitch.py)
        # =====================================================================
        if 'k_view_model' in cerebros and 'k_view_scaler' in cerebros:
            X_scaled_view = cerebros['k_view_scaler'].transform(df)
            pred_viewers = cerebros['k_view_model'].predict(X_scaled_view)[0]
            
            # Validar finitud
            if not np.isfinite(pred_viewers):
                pred_viewers = 0
        else:
            pred_viewers = 0
            print("⚠️ Modelo Viewers de Kick no disponible")
        
        # =====================================================================
        # PARTE 2: PREDICCIÓN DE FOLLOWERS (lógica api_ia_twitch_v2.py con LOG)
        # =====================================================================
        if 'k_fol_model' in cerebros:
            # Aplicar LOG a todas las columnas
            df_log = df.copy()
            for col in df_log.columns:
                df_log[f'{col}_log'] = np.log1p(df_log[col].astype(float))
            
            # Seleccionar SOLO las 6 columnas de Kick (sin comments)
            cols_log = [
                'AVG_VIEWERS_D1_log',
                'AVG_VIEWERS_D14_log',
                'HOURS_STREAMED_D1_log',
                'HOURS_STREAMED_D14_log',
                'FOLLOWERS_D14_log',
                'FOLLOWERS_D21_log'
            ]
            cols_disponibles = [c for c in cols_log if c in df_log.columns]
            X_input = df_log[cols_disponibles]
            
            # Predicción
            pred_crecimiento_log = cerebros['k_fol_model'].predict(X_input)[0]
            
            # Limitar logaritmo
            pred_crecimiento_log = np.clip(pred_crecimiento_log, None, 20)
            
            # Convertir de LOG a Real
            crecimiento_real = np.expm1(pred_crecimiento_log)
            if not np.isfinite(crecimiento_real):
                crecimiento_real = 0
            
            followers_d28 = data.FOLLOWERS_D14 + crecimiento_real
            
            # =====================================================================
            # SANITY CHECK INTELIGENTE: Validar crecimiento razonable
            # =====================================================================
            if data.FOLLOWERS_D14 > 0:
                pct_crecimiento = (crecimiento_real / data.FOLLOWERS_D14) * 100
                
                # Limites según tamaño del canal (IGUAL A TWITCH)
                if data.FOLLOWERS_D14 >= 1_000_000:
                    max_pct = 1.0
                elif data.FOLLOWERS_D14 >= 500_000:
                    max_pct = 2.0
                elif data.FOLLOWERS_D14 >= 100_000:
                    max_pct = 5.0
                else:
                    max_pct = 10.0
                
                # Si excede el límite, ajustar
                if pct_crecimiento > max_pct:
                    print(f"⚠️ Crecimiento anómalo detectado: {pct_crecimiento:.2f}%. Limitando a {max_pct}%")
                    crecimiento_real = (data.FOLLOWERS_D14 * max_pct) / 100
                    followers_d28 = data.FOLLOWERS_D14 + crecimiento_real
                    
        else:
            followers_d28 = data.FOLLOWERS_D14
            crecimiento_real = 0
            print("⚠️ Modelo Followers de Kick no disponible")
        
        return {
            "status": "success",
            "plataforma": "Kick",
            "prediccion": {
                "followers_d28": int(followers_d28),
                "avg_viewers_d30": int(pred_viewers),
                "debug": {
                    "followers_d14": int(data.FOLLOWERS_D14),
                    "crecimiento_neto": int(crecimiento_real),
                    "crecimiento_porcentaje": round((crecimiento_real / data.FOLLOWERS_D14 * 100), 2) if data.FOLLOWERS_D14 > 0 else 0
                }
            },
            "input_received": input_dict
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=400,
            detail=f"Error en predicción Kick: {str(e)}"
        )

# =============================================================================
# HEALTH CHECK
# =============================================================================
@app.get("/health")
def health_check():
    """Verifica el estado de los modelos cargados."""
    return {
        "status": "healthy" if cerebros else "degraded",
        "modelos_cargados": list(cerebros.keys()),
        "twitch_followers_ok": 't_fol_model' in cerebros,
        "twitch_viewers_ok": 't_view_model' in cerebros,
        "kick_followers_ok": 'k_fol_model' in cerebros,
        "kick_viewers_ok": 'k_view_model' in cerebros
    }

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    uvicorn.run(
        "api_hibrido:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
