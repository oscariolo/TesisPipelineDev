"""
Módulo 3: Consensus Engine — Votación Mayoritaria y Fleiss' Kappa
=================================================================
Lee los resultados de los 3 modelos (Módulo 2) y el dataset original,
aplica votación por mayoría (k >= 2), calcula Fleiss' Kappa y extrae
una muestra del 5% para validación manual.
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("consensus_engine")


def compute_fleiss_kappa(ratings_matrix: list[list[int]], num_raters: int) -> float:
    N = len(ratings_matrix)
    k = 2
    n = num_raters
    if N == 0: return 0.0

    p_i = [(sum(x**2 for x in row) - n) / (n * (n - 1)) for row in ratings_matrix]
    P_bar = sum(p_i) / N
    p_j = [sum(row[j] for row in ratings_matrix) / (N * n) for j in range(k)]
    P_e_bar = sum(x**2 for x in p_j)

    if P_e_bar == 1: return 1.0
    return (P_bar - P_e_bar) / (1 - P_e_bar)


def mode_or_fallback(categories: list[str]) -> str:
    err_cats = [c for c in categories if c != "NORMAL"]
    if not err_cats: return "NORMAL"
    return Counter(err_cats).most_common(1)[0][0]


def run_consensus(logs_file: Path, models_files: list[Path], output_dir: Path, sample_frac: float) -> None:
    logger.info("Iniciando Motor de Consenso Multimodelo")

    if not logs_file.exists():
        raise FileNotFoundError(f"Falta: {logs_file}")

    # 1. Cargar originales
    logs_data = [json.loads(line) for line in open(logs_file, "r", encoding="utf-8") if line.strip()]
    df_logs = pd.DataFrame(logs_data).set_index("log_id")
    logger.info("%d logs originales cargados.", len(df_logs))

    # 2. Cargar modelos
    dfs_models = []
    for m_file in models_files:
        if not m_file.exists(): continue
        m_data = [json.loads(l) for l in open(m_file, "r", encoding="utf-8") if l.strip()]
        df_m = pd.DataFrame(m_data)
        if not df_m.empty:
            df_m.set_index("log_id", inplace=True)
            dfs_models.append((m_file.stem, df_m))
            logger.info("   -> %s: %d predicciones", m_file.name, len(df_m))

    if len(dfs_models) < 2:
        logger.error("Mínimo 2 modelos para votación.")
        return

    # 3. Join relacional
    df_votes = pd.DataFrame(index=df_logs.index)
    col_is_error, col_category, col_reasoning = [], [], []
    
    for name, df_m in dfs_models:
        df_votes[f"is_error_{name}"] = df_m["is_error"].astype(int)
        df_votes[f"category_{name}"] = df_m["error_category"]
        df_votes[f"reason_{name}"] = df_m["reasoning"]
        col_is_error.append(f"is_error_{name}")
        col_category.append(f"category_{name}")
        col_reasoning.append(f"reason_{name}")

    df_votes.dropna(subset=col_is_error, inplace=True)
    num_raters = len(col_is_error)
    logger.info("%d logs tienen votos completos de %d modelos.", len(df_votes), num_raters)

    # 4. Majority Voting
    df_votes["votes_error"] = df_votes[col_is_error].sum(axis=1)
    df_votes["consensus_is_error"] = (df_votes["votes_error"] >= 2).astype(int)
    df_votes["consensus_category"] = df_votes.apply(lambda row: mode_or_fallback(row[col_category].tolist()), axis=1)

    # 5. Fleiss' Kappa
    ratings_matrix = [[int(num_raters - e), int(e)] for e in df_votes["votes_error"]]
    kappa = compute_fleiss_kappa(ratings_matrix, num_raters)

    # 6. Unir y Formatear
    df_final = df_logs.join(df_votes, how="inner").reset_index()
    cols_order = ["log_id", "timestamp", "method", "status_code", "path", 
                  "consensus_is_error", "consensus_category", "votes_error"] + col_is_error + ["raw_text"] + col_reasoning
    df_final = df_final[[c for c in cols_order if c in df_final.columns]]

    # 7. Exportar
    output_dir.mkdir(parents=True, exist_ok=True)
    df_final.to_json(output_dir / "ground_truth_dataset.jsonl", orient="records", lines=True, force_ascii=False)
    df_final.to_csv(output_dir / "ground_truth_dataset.csv", index=False, encoding="utf-8")
    
    # 8. Muestreo 5%
    try:
        df_sample = df_final.groupby("consensus_is_error", group_keys=False).apply(lambda x: x.sample(frac=sample_frac, random_state=42))
    except ValueError:
        df_sample = df_final.sample(n=max(1, int(len(df_final) * sample_frac)), random_state=42)
    df_sample.to_csv(output_dir / "expert_validation_sample.csv", index=False, encoding="utf-8")

    # Reporte Final
    errores = df_final["consensus_is_error"].sum()
    normales = len(df_final) - errores
    
    reporte = f"""============================================================
REPORTE DE CONSENSO Y MÉTRICAS
============================================================
Total de logs evaluados : {len(df_final)}
Logs etiquetados ERROR  : {errores} ({(errores/len(df_final))*100:.2f}%)
Logs etiquetados NORMAL : {normales} ({(normales/len(df_final))*100:.2f}%)
------------------------------------------------------------
Inter-Annotator Agreement (Fleiss' Kappa): {kappa:.4f}
============================================================
Muestra experta guardada en: expert_validation_sample.csv
"""
    
    # Guardar reporte en archivo de texto
    report_path = output_dir / "consensus_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(reporte)

    # Imprimir en terminal
    print("\n" + reporte)
    logger.info("Reporte de métricas guardado en: %s", report_path.name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", default="pseudo_labeling/data/processed_logs.jsonl")
    parser.add_argument("--models", nargs="+", default=[
        "pseudo_labeling/data/results_model_a.jsonl",
        "pseudo_labeling/data/results_model_b.jsonl",
        "pseudo_labeling/data/results_model_c.jsonl"
    ])
    parser.add_argument("--outdir", default="pseudo_labeling/data")
    parser.add_argument("--sample-frac", type=float, default=0.05)
    args = parser.parse_args()
    
    run_consensus(Path(args.logs), [Path(m) for m in args.models], Path(args.outdir), args.sample_frac)

