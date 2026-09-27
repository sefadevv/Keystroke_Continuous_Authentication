import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt

# Projedeki diğer modülleri içe aktarıyoruz
from data_preprocessing import prepare_chunk, GLOBAL_VOCAB_SIZE
from model import SiameseNetwork_V6, ContrastiveLoss

def evaluate_model(model, test_loader, device):
    model.eval()
    all_distances = []
    all_labels = []

    with torch.no_grad():
        for batch_x1, batch_x2, batch_labels in test_loader:
            if batch_x1.size(0) <= 1: continue
            
            batch_x1, batch_x2 = batch_x1.to(device), batch_x2.to(device)
            emb1, emb2 = model(batch_x1, batch_x2)
            
            dists = F.pairwise_distance(emb1, emb2).cpu().numpy()
            labels = batch_labels.cpu().numpy()
            
            if dists.ndim == 0:
                dists = np.expand_dims(dists, axis=0)
                
            all_distances.extend(dists)
            all_labels.extend(labels)

    all_distances = np.array(all_distances)
    all_labels = np.array(all_labels)

    scores = -all_distances 
    fpr, tpr, thresholds = roc_curve(all_labels, scores)
    roc_auc = auc(fpr, tpr)

    fnr = 1 - tpr
    eer_index = np.nanargmin(np.absolute((fnr - fpr)))
    eer = fpr[eer_index]
    optimal_distance_threshold = -thresholds[eer_index]

    raw_preds_binary = (all_distances < optimal_distance_threshold).astype(int)
    raw_accuracy = accuracy_score(all_labels, raw_preds_binary) * 100

    print("-" * 50)
    print(f"Optimal Eşik Doğruluğu  : %{raw_accuracy:.2f}")
    print(f"ROC-AUC Skoru           : {roc_auc:.4f}")
    print(f"EER (Equal Error Rate)  : %{eer * 100:.2f}")
    print(f"Optimum Eşik (Mesafe)   : {optimal_distance_threshold:.4f}")
    print("-" * 50)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan Cihaz: {device}")

    # Not: Gerçek projede veri yolları argüman veya config dosyası ile alınmalıdır.
    txt_dir = "./dataset/files"
    if not os.path.exists(txt_dir):
        print("Lütfen dataset klasörünü proje dizinine ekleyin.")
        return

    all_txt_files = sorted([os.path.join(txt_dir, f) for f in os.listdir(txt_dir) if f.endswith('.txt')])
    
    test_files = all_txt_files[:2000]
    train_files_pool = all_txt_files[2000:]
    
    print("Test seti hazırlanıyor...")
    global_test_loader = prepare_chunk(test_files)

    model = SiameseNetwork_V6(vocab_size=GLOBAL_VOCAB_SIZE, dropout_rate=0.3).to(device)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    FILES_PER_CHUNK = 1000   
    TOTAL_CHUNKS = len(train_files_pool) // FILES_PER_CHUNK

    print("Eğitim Başlıyor...")
    for chunk_id in range(TOTAL_CHUNKS):
        chunk_files = train_files_pool[chunk_id * FILES_PER_CHUNK : (chunk_id + 1) * FILES_PER_CHUNK]
        chunk_loader = prepare_chunk(chunk_files)
        
        if chunk_loader is None: continue
        
        epochs = 30 if chunk_id == 0 else 5
        model.train()
        
        for epoch in range(epochs):
            for batch_x1, batch_x2, batch_labels in chunk_loader:
                if batch_x1.size(0) <= 1: continue
                
                batch_x1, batch_x2, batch_labels = batch_x1.to(device), batch_x2.to(device), batch_labels.to(device)
                optimizer.zero_grad()
                
                emb1, emb2 = model(batch_x1, batch_x2)
                loss = criterion(emb1, emb2, batch_labels)
                loss.backward()
                optimizer.step()

        print(f"Chunk {chunk_id + 1}/{TOTAL_CHUNKS} Tamamlandı.")

    print("\nModel Değerlendirmesi Yapılıyor...")
    evaluate_model(model, global_test_loader, device)
    
    torch.save(model.state_dict(), "keystroke_v6_final.pth")
    print("Model 'keystroke_v6_final.pth' olarak kaydedildi.")

if __name__ == "__main__":
    main()