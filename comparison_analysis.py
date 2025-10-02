import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def compare_supervised_vs_unsupervised():
    """Supervised vs Unsupervised sistemlerin karşılaştırması"""
    print("📊 SUPERVISED vs UNSUPERVISED KARŞILAŞTIRMA")
    print("=" * 60)
    
    # Supervised sonuçlar (az önce elde ettiklerimiz)
    supervised_results = {
        'accuracy': 94.97,
        'total_samples': 62600,
        'training_time': '21+ hours',  # 3 epoch x 7+ saat
        'categories_detected': 11,
        'precision': {
            'Normal': 0.95,
            'Data_corrupt': 0.95,
            'Data_cut': 0.92,
            'Data_loss': 0.94,
            'Net_disconnect': 0.95,
            'Net_slow': 0.96,
            'Proc_kill': 0.96,
            'Proc_suspend': 0.95,
            'Sys_dead': 0.97,
            'Sys_panic': 0.89,
            'Complex': 0.96
        }
    }
    
    # Unsupervised sonuçlar (önceki mega analiz)
    unsupervised_results = {
        'anomaly_detection': 'Yes',
        'total_samples': 30000,
        'processing_time': '< 30 minutes',
        'categories_analyzed': 11,
        'anomaly_rates': {
            'Data_corrupt': 0.81,
            'Data_cut': 1.00,
            'Data_loss': 0.91,
            'Net_disconnect': 0.55,
            'Net_slow': 0.62,
            'Proc_kill': 0.71,
            'Proc_suspend': 0.83,
            'Sys_dead': 0.77,
            'Sys_panic': 0.90,
            'Complex': 0.72,
            'Normal': 0.41
        }
    }
    
    # Karşılaştırma tablosu
    print("\n🔍 DETAYLI KARŞILAŞTIRMA:")
    print("-" * 80)
    print(f"{'Özellik':<25} {'Supervised':<25} {'Unsupervised':<25}")
    print("-" * 80)
    print(f"{'Doğruluk':<25} {'%94.97':<25} {'Anomali tespiti':<25}")
    print(f"{'Veri Miktarı':<25} {'62,600 örnek':<25} {'30,000+ örnek':<25}")
    print(f"{'Eğitim Süresi':<25} {'21+ saat':<25} {'< 30 dakika':<25}")
    print(f"{'Etiket Gereksinimi':<25} {'Evet (etiketli)':<25} {'Hayır (etiketsiz)':<25}")
    print(f"{'Sınıf Tahminleri':<25} {'11 kesin sınıf':<25} {'Anomali/Normal':<25}")
    print(f"{'Model Karmaşıklığı':<25} {'Yüksek (BERT+NN)':<25} {'Orta (Clustering)':<25}")
    print(f"{'Gerçek Zamanlı':<25} {'Yavaş':<25} {'Hızlı':<25}")
    
    # Kategori bazlı performans karşılaştırması
    plt.figure(figsize=(15, 10))
    
    # Alt grafik 1: Precision (Supervised) vs Anomaly Rate (Unsupervised)
    plt.subplot(2, 2, 1)
    categories = list(supervised_results['precision'].keys())
    supervised_precision = [supervised_results['precision'][cat] for cat in categories]
    unsupervised_rates = [unsupervised_results['anomaly_rates'].get(cat, 0) for cat in categories]
    
    x = np.arange(len(categories))
    width = 0.35
    
    plt.bar(x - width/2, supervised_precision, width, label='Supervised Precision', alpha=0.8, color='skyblue')
    plt.bar(x + width/2, unsupervised_rates, width, label='Unsupervised Anomaly Rate', alpha=0.8, color='lightcoral')
    
    plt.xlabel('Kategoriler')
    plt.ylabel('Skor')
    plt.title('Kategori Bazlı Performans')
    plt.xticks(x, categories, rotation=45, ha='right')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Alt grafik 2: Avantajlar/Dezavantajlar
    plt.subplot(2, 2, 2)
    plt.text(0.05, 0.95, "🎯 SUPERVISED AVANTAJLAR:", transform=plt.gca().transAxes, 
             fontsize=12, fontweight='bold', va='top', color='green')
    
    supervised_pros = [
        "• Kesin sınıf tahminleri (11 kategori)",
        "• Yüksek doğruluk (%94.97)",
        "• Detaylı anomali türü bilgisi",
        "• Güvenilir sınıflandırma"
    ]
    
    for i, pro in enumerate(supervised_pros):
        plt.text(0.05, 0.85 - i*0.08, pro, transform=plt.gca().transAxes, fontsize=10, va='top')
    
    plt.text(0.05, 0.45, "⚠️ SUPERVISED DEZAVANTAJLAR:", transform=plt.gca().transAxes, 
             fontsize=12, fontweight='bold', va='top', color='red')
    
    supervised_cons = [
        "• Çok uzun eğitim süresi (21+ saat)",
        "• Etiketli veri gereksinimi",
        "• Yüksek hesaplama maliyeti",
        "• Yeni kategoriler için yeniden eğitim"
    ]
    
    for i, con in enumerate(supervised_cons):
        plt.text(0.05, 0.35 - i*0.08, con, transform=plt.gca().transAxes, fontsize=10, va='top')
    
    plt.axis('off')
    
    # Alt grafik 3: Unsupervised Avantajlar/Dezavantajlar
    plt.subplot(2, 2, 3)
    plt.text(0.05, 0.95, "⚡ UNSUPERVISED AVANTAJLAR:", transform=plt.gca().transAxes, 
             fontsize=12, fontweight='bold', va='top', color='green')
    
    unsupervised_pros = [
        "• Çok hızlı işlem (< 30 dakika)",
        "• Etiket gerektirmez",
        "• Gerçek zamanlı anomali tespiti",
        "• Düşük hesaplama maliyeti",
        "• Yeni anomalileri keşfedebilir"
    ]
    
    for i, pro in enumerate(unsupervised_pros):
        plt.text(0.05, 0.85 - i*0.08, pro, transform=plt.gca().transAxes, fontsize=10, va='top')
    
    plt.text(0.05, 0.35, "❌ UNSUPERVISED DEZAVANTAJLAR:", transform=plt.gca().transAxes, 
             fontsize=12, fontweight='bold', va='top', color='red')
    
    unsupervised_cons = [
        "• Sadece anomali/normal ayrımı",
        "• Anomali türü bilgisi yok",
        "• Daha az kesin tahminler"
    ]
    
    for i, con in enumerate(unsupervised_cons):
        plt.text(0.05, 0.25 - i*0.08, con, transform=plt.gca().transAxes, fontsize=10, va='top')
    
    plt.axis('off')
    
    # Alt grafik 4: Kullanım senaryoları
    plt.subplot(2, 2, 4)
    plt.text(0.05, 0.95, "🎯 KULLANIM SENARYOLARI:", transform=plt.gca().transAxes, 
             fontsize=12, fontweight='bold', va='top', color='purple')
    
    scenarios = [
        "📊 SUPERVISED İDEAL:",
        "• Detaylı anomali analizi gerekli",
        "• Yeterli etiketli veri var",
        "• Offline analiz yapılacak",
        "• Kesin sınıf bilgisi önemli",
        "",
        "⚡ UNSUPERVISED İDEAL:",
        "• Gerçek zamanlı monitoring",
        "• Hızlı anomali tespiti",
        "• Etiketli veri yok",
        "• Genel anomali tespiti yeterli"
    ]
    
    for i, scenario in enumerate(scenarios):
        color = 'blue' if 'SUPERVISED' in scenario else 'red' if 'UNSUPERVISED' in scenario else 'black'
        weight = 'bold' if any(x in scenario for x in ['SUPERVISED', 'UNSUPERVISED']) else 'normal'
        plt.text(0.05, 0.9 - i*0.08, scenario, transform=plt.gca().transAxes, 
                fontsize=10, va='top', color=color, fontweight=weight)
    
    plt.axis('off')
    
    plt.suptitle('Supervised vs Unsupervised LogBERT Sistemler Karşılaştırması', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/supervised_vs_unsupervised_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Karşılaştırma grafiği kaydedildi: results/supervised_vs_unsupervised_comparison.png")
    
    # Sonuç önerileri
    print("\n🎯 ÖNERİLER:")
    print("=" * 50)
    print("🔹 Detaylı anomali sınıflandırması → Supervised sistemi kullanın")
    print("🔹 Hızlı gerçek zamanlı tespit → Unsupervised sistemi kullanın")
    print("🔹 Hibrit yaklaşım → İlk tarama unsupervised, detay supervised")
    print("🔹 Üretim ortamı → Unsupervised (monitoring) + Supervised (analiz)")

if __name__ == "__main__":
    compare_supervised_vs_unsupervised()
