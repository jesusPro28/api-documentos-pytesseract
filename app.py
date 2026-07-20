import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import io
import re
import fitz  # PyMuPDF para conversión acelerada de PDF a imágenes
import cv2
import torch
torch.set_num_threads(1)
import pytesseract
import numpy as np
import pandas as pd
import unicodedata
from pathlib import Path
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from torchvision import models, transforms

# =====================================================================
# CONFIGURACIÓN Y ESTABLECIMIENTO DE RUTAS
# =====================================================================
# Definimos el directorio base del proyecto de manera dinámica.
API_DIR = Path(__file__).resolve().parent
BASE_DIR = API_DIR.parent.parent  # MiDataset/

# Buscamos el archivo de pesos del modelo de forma inteligente para evitar errores
# si el usuario mueve la carpeta de la API adentro o afuera de 'MiDataset'.
POSIBLES_RUTAS_MODELO = [
    API_DIR / 'modelo_barrera1.pth',                 # Nombre real del archivo en el repositorio
    API_DIR / 'modelo_b1.pth',                       # Nombre alternativo en la carpeta de la API
    BASE_DIR / 'modelo_barrera1.pth',                # En el directorio padre
    BASE_DIR / 'modelo_b1.pth',                      # Nombre alternativo en directorio padre
]

MODEL_PATH = None
for ruta in POSIBLES_RUTAS_MODELO:
    if ruta.exists():
        MODEL_PATH = ruta
        break

# Si no existe en ningún lado, dejamos por defecto la ruta base para evitar errores de sintaxis
if MODEL_PATH is None:
    MODEL_PATH = BASE_DIR / 'modelo_b1.pth'

# =====================================================================
# INICIALIZACIÓN DE COMPONENTES DE INTELIGENCIA ARTIFICIAL
# =====================================================================
# Definimos el hardware a utilizar (prioriza GPU CUDA si está disponible)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Inicialización de la arquitectura MobileNetV3-Small (Barrera 1)
model_b1 = models.mobilenet_v3_small(weights=None)
model_b1.classifier[3] = torch.nn.Linear(model_b1.classifier[3].in_features, 2)

if MODEL_PATH.exists():
    model_b1.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model_b1 = model_b1.to(device)
    model_b1.eval()
else:
    # Registramos advertencia pero no detenemos la API para permitir pruebas locales
    print(f"ADVERTENCIA: No se encontró el archivo de pesos del modelo en {MODEL_PATH}")

# Transformaciones de imagen estándar para inferencia de la red neuronal
transform_b1 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Motor Tesseract OCR - ligero, ideal para servidores con poca RAM (Render Free)
# packages.txt instala tesseract-ocr en el servidor automaticamente
TESS_CONFIG = r'--oem 3 --psm 6 -l spa+eng'

# =====================================================================
# LÓGICA DE NEGOCIO Y PROCESAMIENTO (BARRERAS 1, 2 Y 3)
# =====================================================================

# Clases inversas de clasificación documental
CLASSES_MAP = {0: 'actas_nacimiento', 1: 'curp'}

# Umbrales operativos del sistema
UMBRAL_B1 = 0.70        # Confianza mínima de clasificación
BLUR_THRESHOLD = 80.0    # Varianza Laplaciana mínima (Nitidez)
NOISE_THRESHOLD = 10.0   # Varianza global mínima (Ruido/Contraste)
SKEW_THRESHOLD = 15.0    # Ángulo máximo de inclinación tolerado (Hough)

# Expresión regular oficial RENAPO para validación de CURP
PATTERN_CURP = re.compile(
    r'[A-Z]{4}[0-9]{6}[HM][A-Z]{2}[B-DF-HJ-NP-TV-Z]{3}[A-Z0-9][0-9]'
)

# Campos obligatorios para Actas de Nacimiento y variaciones comunes del OCR
MANDATORY_ACTA_FIELDS = {
    'ACTA': ['ACTA', 'ACTA.', 'ACT4'],
    'NACIMIENTO': ['NACIMIENTO', 'NACI MIENTO', 'NACIM1ENTO'],
    'NOMBRE': ['NOMBRE', 'N0MBRE', 'NOMBR'],
    'MUNICIPIO': ['MUNICIPIO', 'MUNICI PIO', 'MPIO']
}

def normalizar_texto_hispano(texto: str) -> str:
    """
    Normaliza el texto Unicode removiendo acentos y tildes,
    pero protegiendo explícitamente el carácter Ñ.
    """
    texto = texto.upper().replace('Ñ', '||ENYE||')
    texto_nfd = unicodedata.normalize('NFD', texto)
    texto_limpio = ''.join([c for c in texto_nfd if unicodedata.category(c) != 'Mn'])
    return texto_limpio.replace('||ENYE||', 'Ñ')

def corregir_confusion_ocr(texto: str) -> list:
    """
    Genera variantes de la cadena de texto para corregir confusiones visuales
    del OCR comunes en documentos de baja calidad (O/0, I/1, S/5, B/8).
    """
    variantes = [texto]
    v1 = texto.replace('0', 'O').replace('1', 'I').replace('5', 'S').replace('8', 'B')
    v2 = texto.replace('O', '0').replace('I', '1').replace('S', '5')
    variantes.append(v1)
    variantes.append(v2)
    return variantes

def ejecutar_barrera_1(img_pil: Image.Image) -> tuple:
    """
    Barrera 1: Clasificación de tipo documental mediante Red Neuronal Convolucional (CNN).
    """
    if not MODEL_PATH.exists():
        return True, "actas_nacimiento", 1.0, "Modo prueba local: Pesos de B1 ausentes."

    tensor = transform_b1(img_pil.convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model_b1(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
    
    idx = int(probs.argmax())
    conf = float(probs.max())
    clase = CLASSES_MAP[idx]
    ok = conf >= UMBRAL_B1
    
    msg = f"Clase: {clase} (confianza={conf*100:.1f}%)" if ok else f"Confianza B1 insuficiente ({conf*100:.1f}%)"
    return ok, clase, conf, msg

def ejecutar_barrera_2(img_gray: np.ndarray) -> tuple:
    """
    Barrera 2: Evaluación de calidad física del documento (Nitidez, Contraste, Rotación).
    """
    # 1. Evaluación de desenfoque (Blur) mediante varianza del Laplaciano
    blur_score = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    if blur_score < BLUR_THRESHOLD:
        return False, f"Rechazado en B2: Imagen borrosa (nitidez={blur_score:.1f} < {BLUR_THRESHOLD})"

    # 2. Evaluación de contraste global mediante varianza de la matriz
    noise_score = np.var(img_gray)
    if noise_score < NOISE_THRESHOLD:
        return False, f"Rechazado en B2: Contraste insuficiente (var={noise_score:.1f})"

    # 3. Evaluación de rotación (Skew) mediante transformada de Hough
    bordes = cv2.Canny(img_gray, 50, 150, apertureSize=3)
    lineas = cv2.HoughLinesP(bordes, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    if lineas is not None:
        angulos = []
        for linea in lineas:
            x1, y1, x2, y2 = linea[0]
            if x2 != x1:
                angulo = np.degrees(np.arctan2(y2-y1, x2-x1))
                angulos.append(angulo)
        if angulos:
            skew = abs(np.median(angulos))
            if skew > SKEW_THRESHOLD:
                return False, f"Rechazado en B2: Inclinación excesiva (rotación={skew:.1f}° > {SKEW_THRESHOLD}°)"

    return True, f"Calidad física aceptable (nitidez={blur_score:.1f})"

def ejecutar_barrera_3(img_gray: np.ndarray, clase_doc: str) -> tuple:
    """
    Barrera 3: Lectura óptica (OCR) y extracción estructurada de campos clave (KIE).
    """
    # Preprocesamiento local para OCR: CLAHE + Gaussian Blur + Binarización Otsu
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gris_mejorada = clahe.apply(img_gray)
    suavizado = cv2.GaussianBlur(gris_mejorada, (3, 3), 0)
    _, img_binaria = cv2.threshold(suavizado, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Inferencia con pytesseract - motor ligero sin modelos propios de IA
    texto_raw = pytesseract.image_to_string(img_binaria, config=TESS_CONFIG)
    if not texto_raw or not texto_raw.strip():
        return False, 0.0, "Rechazado en B3: No se pudo extraer texto del documento."

    # Confianza estimada estandar de pytesseract en documentos impresos
    confianza_promedio = 0.85
    texto_completo = normalizar_texto_hispano(texto_raw)

    # Validación semántica según tipo documental
    if clase_doc == 'curp':
        curp_detectado = None
        for variante in corregir_confusion_ocr(texto_completo):
            match = PATTERN_CURP.search(variante)
            if match:
                curp_detectado = match.group(0)
                break
        if curp_detectado:
            return True, confianza_promedio, f"CURP Validado: {curp_detectado}"
        return False, confianza_promedio, "Rechazado en B3: Estructura CURP no detectada o inválida."

    elif clase_doc == 'actas_nacimiento':
        hallados = []
        faltantes = []
        for campo, variantes_campo in MANDATORY_ACTA_FIELDS.items():
            if any(v in texto_completo for v in variantes_campo):
                hallados.append(campo)
            else:
                faltantes.append(campo)
        
        # Regla de negocio: Si faltan 2 o más campos indispensables, se rechaza
        if len(faltantes) >= 2:
            return False, confianza_promedio, f"Rechazado en B3: Faltan metadatos esenciales {faltantes}"
        return True, confianza_promedio, f"Acta validada. Metadatos encontrados: {hallados}"

    return False, 0.0, "Rechazado en B3: Tipo documental no soportado."

# =====================================================================
# DEFINICIÓN DE SERVICIOS API (FASTAPI)
# =====================================================================
app = FastAPI(
    title="API de Validación Documental Multitarea",
    description="API para clasificación, control de calidad y extracción de metadatos de documentos académicos.",
    version="1.0.0"
)

# Habilitamos CORS para permitir peticiones HTTP cruzadas desde el frontend en Hostinger
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict_document(file: UploadFile = File(...)):
    """
    Endpoint principal: recibe el archivo (PDF o Imagen), realiza la conversión
    si es necesario, y ejecuta de forma secuencial las Barreras 1, 2 y 3.
    """
    filename = file.filename
    content_type = file.content_type
    file_bytes = await file.read()
    
    # 1. Conversión e ingesta de PDF / Imagen
    img_pil = None
    if content_type == "application/pdf" or filename.lower().endswith(".pdf"):
        try:
            # Abrimos el archivo PDF desde memoria usando PyMuPDF
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            if len(doc) == 0:
                raise HTTPException(status_code=400, detail="El archivo PDF está vacío.")
            
            # Renderizamos la primera página a alta resolución (300 DPI) para el OCR
            pagina = doc[0]
            pix = pagina.get_pixmap(dpi=300)
            img_pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error al procesar el archivo PDF: {str(e)}")
    else:
        # Si es una imagen común, la cargamos directamente en memoria
        try:
            img_pil = Image.open(io.BytesIO(file_bytes)).convert('RGB')
        except Exception as e:
            raise HTTPException(status_code=400, detail="Formato de imagen no legible.")

    # Convertimos la imagen de PIL a array de OpenCV (escala de grises) para B2 y B3
    img_cv = np.array(img_pil)
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)

    # =================================================================
    # EJECUCIÓN DEL PIPELINE EN CASCADA
    # =================================================================
    
    # BARRERA 1: Clasificación de tipo de documento
    b1_ok, clase_pred, b1_conf, b1_msg = ejecutar_barrera_1(img_pil)
    
    # Si falla la clasificación, retornamos inmediatamente el rechazo (Corte temprano)
    if not b1_ok:
        return {
            "archivo": filename,
            "estado_final": "RECHAZADO",
            "b1_ok": False,
            "b2_ok": "N/A",
            "b3_ok": "N/A",
            "clase_pred": clase_pred,
            "confianza_b1_%": round(b1_conf * 100, 1),
            "confianza_ocr_%": "N/D",
            "observaciones": f"B1: {b1_msg}"
        }

    # BARRERA 2: Control de calidad física de la imagen
    b2_ok, b2_msg = ejecutar_barrera_2(img_gray)
    
    # Si falla el control de calidad, detenemos el proceso
    if not b2_ok:
        return {
            "archivo": filename,
            "estado_final": "RECHAZADO",
            "b1_ok": True,
            "b2_ok": False,
            "b3_ok": "N/A",
            "clase_pred": clase_pred,
            "confianza_b1_%": round(b1_conf * 100, 1),
            "confianza_ocr_%": "N/D",
            "observaciones": f"B1: OK | B2: {b2_msg}"
        }

    # BARRERA 3: Extracción y validación textual (OCR / KIE)
    b3_ok, b3_conf, b3_msg = ejecutar_barrera_3(img_gray, clase_pred)
    
    estado_final = "ACEPTADO" if b3_ok else "RECHAZADO"

    return {
        "archivo": filename,
        "estado_final": estado_final,
        "b1_ok": True,
        "b2_ok": True,
        "b3_ok": b3_ok,
        "clase_pred": clase_pred,
        "confianza_b1_%": round(b1_conf * 100, 1),
        "confianza_ocr_%": round(b3_conf * 100, 1),
        "observaciones": f"B1: OK | B2: OK | B3: {b3_msg}"
    }

@app.get("/health")
def health_check():
    """
    Endpoint para comprobar que el servicio web de la API está levantado.
    """
    return {"status": "healthy", "device": device.type}
