import json
import re

def re_sub_variables(text):
    """
    Abstrae variables dinamicas (IPs, IDs, Timestamps, Numeros) 
    para generar el template del log sin ruido sintactico.
    """
    # Reemplazar IPs
    text = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '<IP>', text)
    # Reemplazar UUIDs / IDs largos
    text = re.sub(r'\b[a-fA-F0-9]{8}(-[a-fA-F0-9]{4}){3}-[a-fA-F0-9]{12}\b', '<UUID>', text)
    # Reemplazar numeros/puertos/latencias
    text = re.sub(r'\b\d+\b', '<NUM>', text)
    return text

def determine_error_label(log_line):
    """
    Determina si un log representa un error y asigna su categoria taxonomica.
    """
    line_upper = log_line.upper()
    
    # 1. Identificar si es Error (Binary Classification)
    error_keywords = ["ERROR", "FATAL", "CRITICAL", "EXCEPTION", "FAILED", "TIMEOUT", "REFUSED"]
    is_error = any(keyword in line_upper for keyword in error_keywords)
    
    # Check para HTTP 5xx
    if re.search(r'HTTP/...\" 5\d\d', log_line) or " 500 " in log_line:
        is_error = True

    # 2. Asignar Categoria Taxonomica
    category = "NORMAL"
    if is_error:
        if "TIMEOUT" in line_upper or "TIME OUT" in line_upper:
            category = "TIMEOUT_ERROR"
        elif "POSTGRES" in line_upper or "MYSQL" in line_upper or "DB" in line_upper or "SQL" in line_upper:
            category = "DATABASE_ERROR"
        elif "SSH" in line_upper or "PASSWORD" in line_upper or "AUTH" in line_upper:
            category = "AUTHENTICATION_ERROR"
        elif "KAFKA" in line_upper or "BROKER" in line_upper or "CONSUMER" in line_upper:
            category = "MESSAGING_KAFKA_ERROR"
        elif "MEMORY" in line_upper or "OOM" in line_upper or "HEAP" in line_upper:
            category = "RESOURCE_EXHAUSTION"
        elif "HTTP" in line_upper or "500" in line_upper or "REST" in line_upper:
            category = "HTTP_SERVICE_FAILURE"
        else:
            category = "GENERIC_SYSTEM_ERROR"
            
    return is_error, category

def parse_log_line(log_line, service_name="university-service"):
    """
    Parsea una linea de log cruda al esquema JSON unificado para el SLM.
    """
    is_error, category = determine_error_label(log_line)
    
    # Detectar nivel de log
    level = "INFO"
    for lvl in ["INFO", "DEBUG", "WARN", "WARNING", "ERROR", "FATAL", "CRITICAL"]:
        if lvl in log_line.upper():
            level = lvl if lvl != "WARNING" else "WARN"
            break
            
    template_msg = re_sub_variables(log_line.strip())
    
    parsed_record = {
        "service": {
            "name": service_name,
            "environment": "production"
        },
        "log_data": {
            "level": level,
            "template": template_msg,
            "raw_message": log_line.strip()
        },
        "ground_truth_label": {
            "is_error": is_error,
            "error_category": category,
            "severity": "HIGH" if is_error else "NONE"
        }
    }
    return parsed_record

# --- EJECUCIÓN CON ARCHIVO DE LOGS ---
def process_log_dataset(input_file_path, output_json_path):
    dataset_parsed = []
    
    with open(input_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.strip(): # Ignorar lineas vacias
                parsed_item = parse_log_line(line)
                dataset_parsed.append(parsed_item)
                
    with open(output_json_path, 'w', encoding='utf-8') as f_out:
        json.dump(dataset_parsed, f_out, indent=2, ensure_ascii=False)
        
    print(f"✅ Procesamiento completado. Total de logs procesados: {len(dataset_parsed)}")
    print(f"📁 Dataset exportado en: {output_json_path}")

# Ejemplo de uso:
process_log_dataset("logs/SSH.log", "dataset_slm_procesado(1).json")
