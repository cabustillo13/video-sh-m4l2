# Pipelines de Audio, WER y Demos de Evaluación

El objetivo es ilustrar flujos de trabajo prácticos para **Reconocimiento Automático del Habla (ASR)**, los efectos de la **tokenización** y el **resumen de audio a texto**.

## Demos

* **Video 1: Comparación de Tokenización**

  **Objetivo:** Observar cómo distintos esquemas de tokenización (Whisper vs Wav2Vec2) afectan la salida del ASR y calcular la **Tasa de Error de Palabras (WER)**.

* **Video 2: Pipeline Audio -> Texto -> Resumen**

  **Objetivo:** Demostrar un pipeline liviano que transcribe audio y lo resume utilizando un LLM preentrenado.

---

## Configuración

1. **Abrí una terminal** en este repositorio.

2. **Creá y activá un entorno virtual**:

   **Usando venv (Linux/macOS/Windows PowerShell):**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate    # Unix/macOS
   .\.venv\Scripts\activate     # Windows PowerShell
   ```

   **O usando conda:**

   ```bash
   conda create -n m4l2 python=3.10
   conda activate m4l2
   ```

3. **Instalá las dependencias**:

   ```bash
   pip install -r requirements.txt
   ```

---

## Ejecución de las Demos

### Video 1: Comparación de Tokenización

```bash
python video-1-tokenization/tokenization_comparison.py
```

* Observa las diferencias de transcripción entre Whisper y Wav2Vec2.
* Calcula la **Tasa de Error de Palabras (WER)** entre las salidas.
* Imprime el conteo de tokens para cada transcripción.

### Video 2: Pipeline Audio -> Texto -> Resumen

```bash
python video-2-summarization/audio_summary.py
```

* Carga una muestra de audio.
* Ejecuta ASR para generar la transcripción.
* Resume la transcripción utilizando un LLM preentrenado.
* Imprime el resumen resultante.

---

## Resolución de Problemas

* **ModuleNotFoundError**: Asegurate de haber instalado todas las dependencias dentro del entorno virtual activado.
* **Archivo de audio no encontrado**: Verificá la ruta a tus archivos `.wav`.
