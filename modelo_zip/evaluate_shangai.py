import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
from pathlib import Path
import scipy.io as sio

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

def get_true_count_from_mat(mat_path):
    data = sio.loadmat(mat_path)
    # Estrutura típica: data['image_info'][0,0][0,0][0] contém as coordenadas (N,2)
    coords = data['image_info'][0,0][0,0][0]
    return len(coords)

def find_ground_truth(image_path):
    """
    Dado o caminho de uma imagem no formato:
        .../part_[AB]/test_data/images/IMG_X.jpg
    ou  .../part_[AB]/train_data/images/IMG_X.jpg
    retorna o caminho para o ficheiro .mat correspondente em:
        .../part_[AB]/test_data/ground-truth/GT_IMG_X.mat
    ou  .../part_[AB]/train_data/ground-truth/GT_IMG_X.mat
    """
    # O diretório da imagem (ex: .../part_A/test_data/images)
    img_dir = image_path.parent
    # O diretório acima (ex: .../part_A/test_data)
    parent_dir = img_dir.parent
    # O nome do ficheiro ground truth
    gt_filename = f"GT_{image_path.stem}.mat"

    # Caminho esperado: parent_dir / "ground-truth" / gt_filename
    candidate = parent_dir / "ground-truth" / gt_filename
    if candidate.exists():
        return candidate
    else:
        # Se não existir, tenta também sem o prefixo GT_ (caso raro)
        alt_candidate = parent_dir / "ground-truth" / f"{image_path.stem}.mat"
        if alt_candidate.exists():
            return alt_candidate
    return None

def evaluate_shanghaitech(process_test=True, process_train=False):
    """
    process_test: se True, processa as imagens de test_data
    process_train: se True, processa as imagens de train_data
    """
    base_dir = Path("resources")
    model_path = Path("model/zip_n_model_quant.onnx")

    if (base_dir / "ShanghaiTech").exists():
        base_dir = base_dir / "ShanghaiTech"
        print(f"Usando diretório base: {base_dir}")

    if not model_path.exists():
        print(f"Erro: modelo não encontrado em {model_path}")
        return

    session = ort.InferenceSession(str(model_path))

    # Pastas a processar conforme os parâmetros
    data_folders = []
    if process_test:
        data_folders.append("test_data")
    if process_train:
        data_folders.append("train_data")

    if not data_folders:
        print("Nenhum conjunto de dados selecionado para processamento.")
        return

    image_paths = []
    for part in ["part_A", "part_B"]:
        part_dir = base_dir / part
        if not part_dir.exists():
            print(f"Aviso: pasta {part_dir} não existe. Ignorando.")
            continue
        for data_folder in data_folders:
            images_dir = part_dir / data_folder / "images"
            if images_dir.exists():
                images = list(images_dir.glob("*.jpg"))
                print(f"Encontradas {len(images)} imagens em {part}/{data_folder}")
                image_paths.extend(images)
            else:
                print(f"Aviso: {images_dir} não existe.")

    if not image_paths:
        print("Nenhuma imagem .jpg encontrada nas pastas especificadas.")
        return

    results = []
    print(f"Total de imagens a processar: {len(image_paths)}")

    for img_path in image_paths:
        gt_path = find_ground_truth(img_path)
        if gt_path is None:
            print(f"Aviso: ground truth não encontrada para {img_path}. Ignorando.")
            continue

        try:
            true_count = get_true_count_from_mat(gt_path)
        except Exception as e:
            print(f"Erro ao ler {gt_path}: {e}")
            continue

        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"Erro ao ler imagem {img_path}")
            continue

        pred_count = predict_count(session, frame)
        abs_error = abs(pred_count - true_count)
        rel_error = (abs_error / true_count * 100) if true_count > 0 else np.nan

        results.append({
            "filename": str(img_path.relative_to(base_dir)),
            "true_count": true_count,
            "predicted_count": pred_count,
            "absolute_error": abs_error,
            "relative_error": rel_error
        })

        print(f"{img_path.name}: real={true_count:3d} | previsto={pred_count:3d} | erro={abs_error:3d}")

    if not results:
        print("Nenhuma imagem foi processada com sucesso.")
        return

    df = pd.DataFrame(results)
    output_csv = Path("resources") / "shanghaitech_predictions.csv"
    df.to_csv(output_csv, index=False)

    # Métricas globais
    true = df["true_count"].values
    pred = df["predicted_count"].values
    abs_err = df["absolute_error"].values

    mae = np.mean(abs_err)
    mse = np.mean((pred - true) ** 2)
    rmse = np.sqrt(mse)

    valid = true > 0
    mape = np.mean(abs_err[valid] / true[valid]) * 100 if valid.any() else np.nan

    percentis = np.percentile(abs_err, [25, 50, 75])

    exact = np.sum(abs_err == 0)
    err5 = np.sum(abs_err <= 5)
    err10 = np.sum(abs_err <= 10)
    err20 = np.sum(abs_err <= 20)

    print("\n" + "="*60)
    print("RESULTADOS DA AVALIAÇÃO NO SHANGHAITECH")
    print("="*60)
    print(f"Total de imagens avaliadas: {len(df)}")
    print(f"Erro absoluto médio (MAE): {mae:.2f}")
    print(f"Erro quadrático médio (MSE): {mse:.2f}")
    print(f"Raiz do erro quadrático médio (RMSE): {rmse:.2f}")
    if not np.isnan(mape):
        print(f"Erro percentual absoluto médio (MAPE): {mape:.2f}%")

    print("\n--- Distribuição dos erros absolutos ---")
    print(f"Mínimo: {np.min(abs_err)}")
    print(f"Percentil 25: {percentis[0]:.2f}")
    print(f"Mediana: {percentis[1]:.2f}")
    print(f"Percentil 75: {percentis[2]:.2f}")
    print(f"Máximo: {np.max(abs_err)}")

    print("\n--- Acertos do modelo ---")
    total = len(df)
    print(f"Previsões exatamente iguais: {exact} ({exact/total*100:.1f}%)")
    print(f"Erro ≤ 5: {err5} ({err5/total*100:.1f}%)")
    print(f"Erro ≤ 10: {err10} ({err10/total*100:.1f}%)")
    print(f"Erro ≤ 20: {err20} ({err20/total*100:.1f}%)")

    # Análise por faixa de contagem real
    bins = [0, 50, 100, 150, 200, 300, 500, 1000]
    labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]
    df["true_bin"] = pd.cut(df["true_count"], bins=bins, labels=labels, right=False)

    bin_stats = df.groupby("true_bin")["absolute_error"].agg(["count", "mean", "std", "min", "max"]).round(2)
    print("\n--- Estatísticas por faixa de contagem real ---")
    print(bin_stats.to_string())

    # Top 5 melhores e piores
    top5 = df.nsmallest(5, "absolute_error")[["filename", "true_count", "predicted_count", "absolute_error"]]
    bottom5 = df.nlargest(5, "absolute_error")[["filename", "true_count", "predicted_count", "absolute_error"]]
    print("\n--- Top 5 imagens com menor erro ---")
    print(top5.to_string(index=False))
    print("\n--- Top 5 imagens com maior erro ---")
    print(bottom5.to_string(index=False))

    print(f"\nResultados detalhados guardados em: {output_csv}")

if __name__ == "__main__":
    # Por padrão, processa apenas test_data. Para processar train_data, muda para True.
    evaluate_shanghaitech(process_test=True, process_train=False)