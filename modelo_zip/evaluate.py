import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
from pathlib import Path

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

def preprocess_image(frame, target_size=(448, 448)):
    image = cv2.resize(frame, target_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = (image - MEAN.reshape(1, 1, 3)) / STD.reshape(1, 1, 3)
    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, axis=0)
    return image

def predict_count(session, frame):
    input_shape = session.get_inputs()[0].shape
    if len(input_shape) == 4 and all(isinstance(dim, int) for dim in input_shape[2:]):
        h, w = input_shape[2], input_shape[3]
    else:
        h, w = 448, 448
    blob = preprocess_image(frame, target_size=(w, h))
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: blob})
    density_map = outputs[0][0][0]
    return int(round(np.sum(density_map)))

def evaluate():
    model_path = Path("model/zip_n_model_quant.onnx")
    resources_dir = Path("resources")
    gt_csv = resources_dir / "ground_truth.csv"

    if not model_path.exists():
        print(f"Erro: modelo não encontrado em {model_path}")
        return
    if not gt_csv.exists():
        print(f"Erro: ficheiro ground_truth.csv não encontrado em {gt_csv}")
        return

    df_gt = pd.read_csv(gt_csv)
    gt_dict = dict(zip(df_gt["filename"], df_gt["count"]))

    session = ort.InferenceSession(str(model_path))

    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif")
    image_files = [f for f in resources_dir.iterdir() if f.suffix.lower() in image_extensions]

    pred_list = []
    true_list = []
    filename_list = []

    print(f"A processar {len(image_files)} imagens...\n")

    for img_path in image_files:
        if img_path.name not in gt_dict:
            print(f"Aviso: {img_path.name} não tem ground truth. Ignorando.")
            continue

        true_count = gt_dict[img_path.name]
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"Erro ao ler {img_path.name}")
            continue

        pred_count = predict_count(session, frame)
        pred_list.append(pred_count)
        true_list.append(true_count)
        filename_list.append(img_path.name)

        print(f"{img_path.name}: real={true_count:3d} | previsto={pred_count:3d} | erro={abs(pred_count - true_count):3d}")

    if not pred_list:
        print("Nenhuma imagem processada.")
        return

    true = np.array(true_list)
    pred = np.array(pred_list)
    errors = pred - true
    abs_errors = np.abs(errors)

    # Métricas básicas
    mae = np.mean(abs_errors)
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)

    # Erro percentual absoluto médio (MAPE) - apenas para valores reais > 0
    mask = true > 0
    if np.any(mask):
        mape = np.mean(abs_errors[mask] / true[mask]) * 100
    else:
        mape = np.nan

    # Estatísticas dos erros absolutos
    min_err = np.min(abs_errors)
    max_err = np.max(abs_errors)
    percentis = np.percentile(abs_errors, [25, 50, 75])

    # Contagem de acertos dentro de limiares
    exact_hits = np.sum(abs_errors == 0)
    hits_5 = np.sum(abs_errors <= 5)
    hits_10 = np.sum(abs_errors <= 10)
    hits_20 = np.sum(abs_errors <= 20)

    # Análise por faixas de contagem real
    bins = [0, 50, 100, 150, 200, 300, 500, 1000]
    labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]
    true_binned = pd.cut(true, bins=bins, labels=labels, right=False)

    df_results = pd.DataFrame({
        "filename": filename_list,
        "true_count": true,
        "predicted_count": pred,
        "absolute_error": abs_errors,
        "true_bin": true_binned
    })

    # Estatísticas por bin
    bin_stats = df_results.groupby("true_bin")["absolute_error"].agg(["count", "mean", "std", "min", "max"]).round(2)

    # Guardar CSV detalhado
    df_results.to_csv(resources_dir / "predictions_detailed.csv", index=False)

    # Apresentar resultados
    print("\n" + "="*60)
    print("RESULTADOS DA AVALIAÇÃO")
    print("="*60)
    print(f"Imagens avaliadas: {len(pred)}")
    print(f"Erro absoluto médio (MAE): {mae:.2f}")
    print(f"Erro quadrático médio (MSE): {mse:.2f}")
    print(f"Raiz do erro quadrático médio (RMSE): {rmse:.2f}")
    if not np.isnan(mape):
        print(f"Erro percentual absoluto médio (MAPE): {mape:.2f}%")
    print("\n--- Distribuição dos erros absolutos ---")
    print(f"Mínimo: {min_err}")
    print(f"Percentil 25: {percentis[0]:.2f}")
    print(f"Mediana (Percentil 50): {percentis[1]:.2f}")
    print(f"Percentil 75: {percentis[2]:.2f}")
    print(f"Máximo: {max_err}")

    print("\n--- Acertos do modelo ---")
    print(f"Previsões exatamente iguais ao real: {exact_hits} ({exact_hits/len(pred)*100:.1f}%)")
    print(f"Previsões com erro ≤ 5: {hits_5} ({hits_5/len(pred)*100:.1f}%)")
    print(f"Previsões com erro ≤ 10: {hits_10} ({hits_10/len(pred)*100:.1f}%)")
    print(f"Previsões com erro ≤ 20: {hits_20} ({hits_20/len(pred)*100:.1f}%)")

    print("\n--- Análise por faixa de contagem real ---")
    print(bin_stats.to_string())

    # Identificar as imagens com menor erro (top 5 mais precisas)
    top5 = df_results.nsmallest(5, "absolute_error")[["filename", "true_count", "predicted_count", "absolute_error"]]
    print("\n--- Top 5 imagens com menor erro ---")
    print(top5.to_string(index=False))

    # Identificar as imagens com maior erro
    bottom5 = df_results.nlargest(5, "absolute_error")[["filename", "true_count", "predicted_count", "absolute_error"]]
    print("\n--- Top 5 imagens com maior erro ---")
    print(bottom5.to_string(index=False))

    print(f"\nRelatório detalhado guardado em: {resources_dir / 'predictions_detailed.csv'}")

if __name__ == "__main__":
    evaluate()