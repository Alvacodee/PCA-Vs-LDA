def main():
    # ==========================================
    # IMPORT LIBRARY
    # ==========================================
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_olivetti_faces
    from sklearn.model_selection import train_test_split
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    import time

    # ==========================================
    # 1. PERSIAPAN DATASET
    # ==========================================
    print("="*50)
    print("1. DATA PREPARATION")
    print("="*50)

    print("Sedang memuat dataset Olivetti Faces...")
    data = fetch_olivetti_faces(shuffle=True, random_state=42)
    X = data.data
    y = data.target
    n_samples, n_features = X.shape

    print(f"Dataset dimuat: {n_samples} gambar, {n_features} fitur (64x64 piksel).")

    # Split Data: 60% Latih, 40% Uji
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )

    print(f"Data Training: {X_train.shape[0]} gambar")
    print(f"Data Testing: {X_test.shape[0]} gambar")

    # ===========================================================
    # 2. EKSPERIMEN UTAMA: EIGENFACES (PCA) VS FISHERFACES (LDA)
    # ===========================================================
    print("\n" + "="*50)
    print("2. MAIN EXPERIMENT: PCA vs LDA")
    print("="*50)

    # A. Eigenfaces (PCA)
    print("Training Eigenfaces (PCA)...")
    n_components_pca = 50 
    pca = PCA(n_components=n_components_pca, whiten=True)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    knn_pca = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    knn_pca.fit(X_train_pca, y_train)
    acc_pca = knn_pca.score(X_test_pca, y_test)
    print(f"-> Akurasi PCA ({n_components_pca} components): {acc_pca*100:.2f}%")

    # B. Fisherfaces (LDA)
    print("Training Fisherfaces (LDA)...")
    lda = LDA(n_components=None) # Otomatis max komponen = C-1 (39)
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)

    knn_lda = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    knn_lda.fit(X_train_lda, y_train)
    acc_lda = knn_lda.score(X_test_lda, y_test)
    print(f"-> Akurasi LDA (Fisherfaces): {acc_lda*100:.2f}%")

    # ================================================
    # 3. EKSPERIMEN TAMBAHAN A: ANALISIS KOMPONEN PCA
    # ================================================
    print("\n" + "="*50)
    print("3. ADDITIONAL EXPERIMENT A: PCA COMPONENTS ANALYSIS")
    print("="*50)
    print("Sedang menguji variasi jumlah komponen PCA...")

    pca_component_range = [5, 10, 20, 30, 40, 50, 75, 100, 150]
    pca_accuracies = []

    for n in pca_component_range:
        pca_temp = PCA(n_components=n, whiten=True)
        X_train_temp = pca_temp.fit_transform(X_train)
        X_test_temp = pca_temp.transform(X_test)
        
        knn_temp = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
        knn_temp.fit(X_train_temp, y_train)
        acc = knn_temp.score(X_test_temp, y_test)
        pca_accuracies.append(acc)
        print(f"   -> n={n}, Accuracy={acc*100:.1f}%")

    # ================================================
    # 4. EKSPERIMEN TAMBAHAN B: ANALISIS KOMPONEN LDA
    # ================================================
    print("\n" + "="*50)
    print("4. ADDITIONAL EXPERIMENT B: LDA COMPONENTS ANALYSIS")
    print("="*50)
    print("Sedang menguji variasi jumlah komponen LDA...")

    # LDA max komponen = 39. Tes dari 1 sampai 39 dengan kelipatan 2.
    lda_component_range = list(range(1, 40, 2)) 
    lda_accuracies = []

    for n in lda_component_range:
        lda_temp = LDA(n_components=n)
        X_train_temp = lda_temp.fit_transform(X_train, y_train)
        X_test_temp = lda_temp.transform(X_test)
        
        knn_temp = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
        knn_temp.fit(X_train_temp, y_train)
        acc = knn_temp.score(X_test_temp, y_test)
        lda_accuracies.append(acc)

        print(f"   -> n={n}, Accuracy={acc*100:.1f}%")

    # ==========================================
    # 5. VISUALISASI
    # ==========================================
    print("\n" + "="*50)
    print("5. VISUALIZATION")
    print("="*50)

    # GAMBAR 1: Visualisasi Eigenfaces (Grayscale)
    plt.figure(figsize=(10, 4))
    plt.suptitle("Top 10 Eigenfaces (PCA) - Reconstructive Features", fontsize=14)
    eigenfaces = pca.components_[:10].reshape((10, 64, 64))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(eigenfaces[i], cmap='gray')
        plt.axis('off')
    plt.show()

    # GAMBAR 2: Visualisasi Fisherfaces (Color Map)
    fisherfaces = lda.scalings_.T[:10].reshape((10, 64, 64))
    plt.figure(figsize=(10, 4))
    plt.suptitle("Top 10 Fisherfaces (LDA) - Discriminative Features", fontsize=14)
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(fisherfaces[i], cmap='jet') 
        plt.axis('off')
    plt.show()

    # GAMBAR 3: Grafik Garis Pengaruh Jumlah Komponen PCA
    plt.figure(figsize=(8, 5))
    plt.plot(pca_component_range, pca_accuracies, marker='o', linestyle='-', color='b', linewidth=2)
    plt.title('Effect of Number of Principal Components on Accuracy', fontsize=14)
    plt.xlabel('Number of Eigenfaces (Components)', fontsize=12)
    plt.ylabel('Recognition Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    max_pca_acc = max(pca_accuracies)
    max_pca_idx = pca_component_range[pca_accuracies.index(max_pca_acc)]
    plt.scatter(max_pca_idx, max_pca_acc, color='red', s=50, zorder=5) 
    plt.text(max_pca_idx, max_pca_acc + 0.015, f"Max: {max_pca_acc*100:.1f}%", 
            ha='center', va='bottom', fontsize=10, fontweight='bold', color='red')
    plt.ylim(min(pca_accuracies)-0.05, 1.1) 
    plt.show()

    # GAMBAR 4: Grafik Garis Pengaruh Jumlah Komponen LDA (HANYA LDA)
    plt.figure(figsize=(8, 5))
    plt.plot(lda_component_range, lda_accuracies, marker='s', linestyle='-', color='g', linewidth=2)
    plt.title('Effect of Number of Fisherfaces (LDA) on Accuracy', fontsize=14)
    plt.xlabel('Number of Fisherfaces (Components)', fontsize=12)
    plt.ylabel('Recognition Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    max_lda_acc = max(lda_accuracies)
    max_lda_idx = lda_component_range[lda_accuracies.index(max_lda_acc)]
    plt.scatter(max_lda_idx, max_lda_acc, color='red', s=50, zorder=5)
    plt.text(max_lda_idx, max_lda_acc + 0.015, f"Max: {max_lda_acc*100:.1f}%", 
            ha='center', va='bottom', fontsize=10, fontweight='bold', color='red')
    plt.ylim(min(lda_accuracies)-0.05, 1.1) 
    plt.show()

    # GAMBAR 5: Grafik Garis Efisiensi LDA vs PCA (Perbandingan)
    plt.figure(figsize=(8, 5))
    plt.plot(lda_component_range, lda_accuracies, marker='s', linestyle='-', color='g', linewidth=2, label='LDA (Fisherfaces)')
    # Plot PCA sebagai pembanding (range < 40)
    pca_range_short = [x for x in pca_component_range if x <= 40]
    pca_acc_short = pca_accuracies[:len(pca_range_short)]
    plt.plot(pca_range_short, pca_acc_short, marker='o', linestyle='--', color='b', linewidth=2, label='PCA (Eigenfaces)')
    plt.title('Efficiency Comparison: LDA vs PCA Components', fontsize=14)
    plt.xlabel('Number of Components (Features)', fontsize=12)
    plt.ylabel('Recognition Accuracy', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # GAMBAR 6: Bar Chart Perbandingan Akhir
    plt.figure(figsize=(6, 5))
    final_methods = ['Eigenfaces\n(PCA n=50)', 'Fisherfaces\n(LDA)']
    final_scores = [acc_pca, acc_lda]
    colors = ['#4c72b0', '#55a868']
    bars = plt.bar(final_methods, final_scores, color=colors, width=0.6)
    plt.title('Final Accuracy Comparison', fontsize=14)
    plt.ylim(0, 1.15)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, 
                f"{yval*100:.1f}%", ha='center', va='bottom', fontsize=12, fontweight='bold')
    plt.show()


if __name__ == "__main__":
    main()
