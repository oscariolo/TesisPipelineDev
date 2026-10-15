# CONTEXTO DEL PROYECTO DE TESIS Y ESTADO ACTUAL

> **Instrucciones para Claude / LLM receptor:**  
> Este documento contiene el contexto integral del proyecto de tesis, su arquitectura técnica, el estado del código y los hallazgos/decisiones más recientes para que puedas continuar asistiendo al usuario sin pérdida de contexto.

---

## 1. Información General del Proyecto

* **Título / Propósito:** Pipeline de análisis por lotes de logs técnicos para la **DTIC (Dirección de Tecnologías de la Información y Comunicación) de la Universidad de Cuenca**, utilizando Small Language Models (SLMs) locales para clasificar eventos como error o no-error, junto con la construcción de datasets, fine-tuning con LoRA y evaluación formal de modelos.
* **Entorno Técnico y Hardware Heterogéneo (2 Máquinas):**
  * **Sistema Operativo:** Linux (Ubuntu).
  * **Máquina 1 (Usuario actual):** GPU **NVIDIA** con aceleración **CUDA** (dependencias gestionadas con `scripts/requirements.nvidia.txt`, dispositivo `cuda:0` / `--hf-device cuda:0`).
  * **Máquina 2 (Compañero de tesis):** GPU **AMD** con aceleración **ROCm** (`torch==2.13.0+rocm7.2`, dependencias en `scripts/requierements.txt`).
  * **Python:** Virtualenv en `scripts/.venv` (Python 3.13).
  * **Motor de inferencia local:** **Ollama** (puerto `11434`) y **Hugging Face Transformers**.
  * **Base de datos vectorial:** **Milvus Lite** (`scripts/embeddings/log_embeddings.db`).
  * **Estrategia de paralelización:** Disponer de dos máquinas con GPU acelerada (una NVIDIA y otra AMD) permite repartir los 3 modelos del pipeline de Ground Truth (o tareas de fine-tuning) entre ambos equipos para correr en paralelo durante el fin de semana.
  * **Directorio de trabajo:** Todo el código ejecutable vive dentro de `scripts/`. Todos los comandos se ejecutan desde `scripts/`.

---

## 2. Arquitectura del Pipeline Principal

El proyecto consta de tres grandes componentes dentro de `scripts/`:

1. **Pipeline de Inferencia en Tiempo Real / Batch (`scripts/log_analysis/` y `main.py`):**
   * **Modo Generativo (`--mode generative`):** Envía lotes de logs a un LLM (vía Ollama o Hugging Face) para clasificación binaria y extracción de causa raíz.
   * **Modo Embedding / Cache Semántico (`--mode embedding`):** Consulta primero una base vectorial en Milvus Lite buscando logs idénticos/similares previamente analizados. Si hay cache hit, reutiliza la clasificación; si no, hace fallback al LLM y guarda la nueva inferencia.
   * **Evaluación y Métricas (`--evaluate`):** Compara los resultados contra el Ground Truth calculando ROC AUC, Precision, Recall, F1-Score y matriz de confusión en `analysis/<model>/`.
   * **Dashboard:** Visualización interactiva en Plotly Dash (`dashboard/app.py`).

2. **Fine-Tuning con LoRA (`scripts/training/`):**
   * Ajuste fino de modelos SLM (ej. LLaMA 3.2 1B / 3B) sobre datasets de logs etiquetados.

3. **Módulo de Creación de Ground Truth (`scripts/pseudo_labeling/`):**
   Pipeline de 3 etapas secuenciales basado en consenso multimodelo:
   * **Etapa 1 (`log_preprocessor.py`):** Parsea logs crudos y genera `processed_logs.jsonl` con un `log_id` secuencial (`LOG-XXXXX`).
   * **Etapa 2 (`batch_runner.py`):** Envía cada log a **3 modelos independientes en Ollama** (`results_model_a.jsonl`, `results_model_b.jsonl`, `results_model_c.jsonl`). Cuenta con sistema de reintentos y reanudación automática (`resume`).
   * **Etapa 3 (`consensus_engine.py`):** Aplica votación mayoritaria ($k \ge 2$), calcula acuerdo entre anotadores (**Fleiss' Kappa**), exporta el dataset final de ground truth y extrae una muestra estratificada del 5% para validación de expertos.

---

## 3. Hallazgo Crítico Reciente: Los Nuevos Logs de la DTIC

### A. Situación Anterior vs. Situación Actual
* **Antes:** El pipeline de Ground Truth fue diseñado y probado con un dataset sintético/externo de servidor web (`webServer.log` en formato Apache/Nginx Combined Log: `IP - - [date] "GET /path" STATUS BYTES`). Eran 10.001 líneas limpias donde 1 línea = 1 petición HTTP con código de estado (200, 404, 500).
* **Ahora:** El tutor de tesis entregó los logs **reales** del microservicio de administración de usuarios de la Universidad de Cuenca: `ms-users` (`ec.edu.ucuenca.msusersucuenca`).
  * Total de archivos: **309 archivos** nombrados por fecha/hora (`logFile.YYYY-MM-DD.HH.log`).
  * Tamaño promedio: **~50 MB por archivo** (total: **~15 GB** de logs).

### B. Diagnóstico Técnico de los Logs de `ms-users`
Al analizar a fondo el archivo `logFile.2026-10-06.21.log` y una muestra de los 309 archivos, se descubrió la causa del enorme peso:
1. **Nivel TRACE en Spring Security:** El microservicio tiene activado el logging a nivel `TRACE` para Spring Security (`org.springframework.security`), imprimiendo cada paso de una cadena de 15 filtros por petición.
2. **Sondeos periódicos de Kubernetes a Actuator:** Cada 5 a 15 segundos, pods internos (`10.16.0.189`, `10.16.0.192`) y servidores de monitoreo hacen sondeos a `/actuator/health` y `/actuator/prometheus`.
3. **El efecto bola de nieve:**
   * `/actuator` no tiene acceso anónimo permitido.
   * Spring Security bloquea el sondeo, lanza `AccessDeniedException` e **imprime un stack trace de 115 líneas por cada sondeo**.
   * Luego redirige a `/login`, ejecutando de nuevo los 15 filtros TRACE y los manejadores de recursos de Spring MVC.
4. **Impacto en volumen:**
   * En 1 hora hay ~215 sondeos de salud. Cada uno genera ~160 líneas.
   * **El 99.8% - 99.9% del contenido de los 15 GB es ruido repetitivo de infraestructura** (stack traces de Actuator y trazas internas de filtros).
   * Solo el **0.07% - 0.1%** son eventos reales de negocio de la universidad.
   * En los 309 archivos hay aproximadamente entre **50.000 y 60.000 logs reales de negocio**, no millones.

### C. Los Logs Reales de la Universidad (Datos Valiosos para la Tesis)
Al filtrar el ruido de infraestructura, aparecen los verdaderos eventos de negocio del microservicio de usuarios:
* **Errores (`ERROR`):**
  * `AdUserOutAdappter : Password does not meet security policies` (política de contraseñas de Active Directory).
  * `UsuarioAs400Service : User not found with personal identification as400: X` (fallo de consulta a la base de datos AS400).
  * `UsuarioOracleService : User not found with personal identification: X` (fallo en Oracle).
  * `AdUserOutAdappter : User not found in AD: ...`
  * `GoogleUserOutAdapter : Usuario ... no encontrado en Google Workspace`
* **Advertencias (`WARN`):**
  * `LdapEstudianteOutAdapter : Usuario no encontrado en la rama estudiante de LDAP con mail: ...`
* **Operaciones exitosas (`INFO` / `DEBUG`):**
  * `AdUser password changed successfully`
  * `Contraseña cambiada en la rama estudiante de LDAP`
  * `Completed 200 OK`

---

## 4. Reglas y Decisiones de Diseño Acordadas

1. **Filtrado por Endpoints y Trazas, NUNCA por IP:**
   * Las IPs `10.16.0.x` corresponden a los pods de Ingress Controller / Reverse Proxy de Kubernetes. Como Tomcat no siempre tiene el `RemoteIpValve` con `X-Forwarded-For`, las peticiones legítimas de usuarios también entran con esa IP.
   * **Regla:** Se filtran las rutas `/actuator/**`, los redirects a `/login` causados por Actuator y las trazas `TRACE` internas de `FilterChainProxy`, pero **no** se descarta tráfico basándose en la IP de red.

2. **Obligatoriedad de 3 Modelos para el Consenso (Votación):**
   * El usuario estableció explícitamente: **"Si o si debemos usar 3 modelos, debido a que se realiza el voting"**.
   * El motor de consenso requiere mínimo 3 jueces para tener mayoría estricta ($k \ge 2$) y calcular Fleiss' Kappa.

3. **Modo Fast (`--fast`) vs. Modo Normal:**
   * En la corrida previa de 10.001 logs, el modo normal tardó **~19 horas** en total porque cada modelo generaba un JSON detallado con categoría, confianza y razonamiento (~150-200 tokens generados por respuesta).
   * Para ~50.000 logs reales, el modo normal demoraría ~100 horas.
   * Se diseñó el **Modo Fast**:
     * Prompt minimalista que pide únicamente responder con la palabra `true` o `false`.
     * `temperature: 0` (decoding codicioso sin sampling) y `num_predict: 5` (máximo 5 tokens generados).
     * Pasa de tardar 2–4 segundos por log a **~0.15 segundos por log**.
     * Los ~50.000 logs se pueden procesar en **~12 a 16 horas en total** (o ~6 a 8 horas si se dividen modelos entre 2 computadoras).

4. **Procesamiento Log-por-Log (No Batches):**
   * Se mantendrá el análisis **log por log** para la etapa de Ground Truth.
   * Razón: Los SLMs de 7B/8B pierden alineación de índices al evaluar listas en batch (ej. enviar 10 logs y que el modelo devuelva 9 respuestas desordenadas). Además, log por log permite reanudar (`resume`) de inmediato en caso de interrupciones o apagones.

5. **Candidatos a Modelos para el Trío de Votación:**
   * `llama3.1:8b` (rápido, ~0.89 logs/s normal, ~7 logs/s en fast)
   * `mistral:7b` (rápido, ~0.87 logs/s normal, ~7 logs/s en fast)
   * El tercer modelo: `qwen2.5:7b` demostró ser 4x más lento (0.22 logs/s) por su thinking interno. Se planteó evaluar sustituirlo en el trío por un modelo más ligero como `llama3.2:3b` o `gemma2:2b`, o mantener `qwen2.5:7b` ejecutándolo en una máquina dedicada en modo fast.

6. **Regla de Trabajo con el Asistente:**
   * **NUNCA modificar archivos ni código sin confirmación previa del usuario.** Siempre proponer y explicar antes de aplicar cambios.

---

## 5. Próximos Pasos Técnicos Pendientes

1. **Script de Pre-filtrado y Limpieza de los 309 archivos:**
   * Crear un script en Python que procese los 309 archivos de 50 MB, elimine las cascadas de Actuator y filtros TRACE de Spring Security, y agrupe las líneas de stack trace multilínea (`\tat ...`) con su evento padre.
   * Guardar el dataset limpio y compacto (estimado en < 100 MB y ~50.000 eventos).

2. **Adaptación de `log_preprocessor.py`:**
   * Modificar el parser regex para que lea formato Logback de Spring Boot:
     `Timestamp LEVEL PID --- [thread] Logger : Message`
     en lugar del regex antiguo de Apache HTTP.

3. **Ejecución del Batch Runner en Modo Fast con los 3 Modelos:**
   * Anotar los ~50.000 logs limpios con los 3 modelos seleccionados vía Ollama usando `--fast`.
   * Ejecutar el `consensus_engine.py` para generar el `ground_truth_dataset.jsonl` definitivo.

