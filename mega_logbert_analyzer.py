import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import gc
import os
import time
from pathlib import Path
import pickle
import json
warnings.filterwarnings('ignore')

class MegaDataLogBERT:
    """Çok büyük HDFS trace verisi için optimize edilmiş LogBERT"""
    
    def __init__(self, model_name='bert-base-uncased', max_length=64, batch_size=8):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Kullanılan cihaz: {self.device}")
        
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        self.max_length = max_length
        self.batch_size = batch_size
        self.scaler = StandardScaler()
        
        # Büyük veri parametreleri
        self.sample_size = 50000  # İşlenecek maksimum log sayısı
        self.chunk_size = 5000    # Her seferinde işlenecek chunk
        
    def load_trace_data(self, data_dir="data/HDFS_big.log"):
        """Trace veri setini yükle ve analiz et"""
        print("📊 MEGA TRACE VERİ SETİ ANALİZİ BAŞLADI")
        print("=" * 60)
        
        data_path = Path(data_dir)
        preprocessed_path = data_path / "preprocessed"
        tracebench_path = data_path / "tracebench"
        
        # Preprocessed verileri analiz et
        print("🔍 Preprocessed veriler analiz ediliyor...")
        
        normal_file = preprocessed_path / "normal_trace.csv"
        failure_file = preprocessed_path / "failure_trace.csv"
        
        normal_data = None
        failure_data = None
        
        try:
            # Normal trace verilerini yükle (sample)
            print("📖 Normal trace veriler yükleniyor...")
            normal_data = pd.read_csv(normal_file, nrows=self.sample_size//2)
            print(f"✅ Normal trace: {len(normal_data):,} kayıt yüklendi")
            
            # Failure trace verilerini yükle (sample) 
            print("📖 Failure trace veriler yükleniyor...")
            failure_data = pd.read_csv(failure_file, nrows=self.sample_size//2)
            print(f"✅ Failure trace: {len(failure_data):,} kayıt yüklendi")
            
        except Exception as e:
            print(f"❌ Preprocessed veriler yüklenemedi: {str(e)}")
            return None, None
        
        return normal_data, failure_data
    
    def explore_tracebench_data(self, data_dir="data/HDFS_big.log"):
        """Tracebench klasörünü keşfet"""
        print("\n🚀 TRACEBENCH VERİLERİ KEŞFEDİLİYOR")
        print("=" * 50)
        
        tracebench_path = Path(data_dir) / "tracebench"
        
        # Anomali türlerini kategorize et
        anomaly_categories = {
            'Data_corrupt': [],
            'Data_cut': [],
            'Data_loss': [],
            'Net_disconnect': [],
            'Net_slow': [],
            'Proc_kill': [],
            'Proc_suspend': [],
            'Sys_dead': [],
            'Sys_panic': [],
            'Complex': [],
            'Normal': []
        }
        
        # Tüm klasörleri tara
        for folder in tracebench_path.iterdir():
            if folder.is_dir():
                folder_name = folder.name
                
                if folder_name.startswith('AN_Data_corrupt'):
                    anomaly_categories['Data_corrupt'].append(folder_name)
                elif folder_name.startswith('AN_Data_cut'):
                    anomaly_categories['Data_cut'].append(folder_name)
                elif folder_name.startswith('AN_Data_loss'):
                    anomaly_categories['Data_loss'].append(folder_name)
                elif folder_name.startswith('AN_Net_disconnect'):
                    anomaly_categories['Net_disconnect'].append(folder_name)
                elif folder_name.startswith('AN_Net_slow'):
                    anomaly_categories['Net_slow'].append(folder_name)
                elif folder_name.startswith('AN_Proc_kill'):
                    anomaly_categories['Proc_kill'].append(folder_name)
                elif folder_name.startswith('AN_Proc_suspend'):
                    anomaly_categories['Proc_suspend'].append(folder_name)
                elif folder_name.startswith('AN_Sys_dead'):
                    anomaly_categories['Sys_dead'].append(folder_name)
                elif folder_name.startswith('AN_Sys_panic'):
                    anomaly_categories['Sys_panic'].append(folder_name)
                elif folder_name.startswith('COM_'):
                    anomaly_categories['Complex'].append(folder_name)
                elif folder_name.startswith('NM_'):
                    anomaly_categories['Normal'].append(folder_name)
        
        # İstatistikleri yazdır
        print("📊 ANOMALİ TÜRÜ İSTATİSTİKLERİ:")
        total_scenarios = 0
        for category, folders in anomaly_categories.items():
            count = len(folders)
            total_scenarios += count
            if count > 0:
                print(f"  {category.replace('_', ' ').title()}: {count:,} senaryo")
        
        print(f"\nToplam senaryo sayısı: {total_scenarios:,}")
        
        return anomaly_categories
    
    def sample_diverse_scenarios(self, anomaly_categories, samples_per_category=2):
        """Her kategoriden örnekler seç"""
        print(f"\n🎯 Her kategoriden {samples_per_category} örnek seçiliyor...")
        
        selected_scenarios = {}
        total_selected = 0
        
        for category, folders in anomaly_categories.items():
            if folders and len(folders) > 0:
                # Rastgele örnekle
                n_samples = min(samples_per_category, len(folders))
                selected = np.random.choice(folders, n_samples, replace=False)
                selected_scenarios[category] = list(selected)
                total_selected += n_samples
                print(f"  {category}: {n_samples} senaryo seçildi")
        
        print(f"✅ Toplam {total_selected} senaryo seçildi")
        return selected_scenarios
    
    def load_scenario_logs(self, scenario_path):
        """Bir senaryonun log dosyalarını yükle"""
        logs = []
        
        try:
            scenario_dir = Path(scenario_path)
            
            # event.csv dosyasından log verileri al
            event_file = scenario_dir / "event.csv"
            if event_file.exists():
                try:
                    df = pd.read_csv(event_file, nrows=1000)  # İlk 1000 kayıt
                    if 'Description' in df.columns and 'OpName' in df.columns:
                        # OpName ve Description'ı birleştirerek log mesajı oluştur
                        log_messages = []
                        for _, row in df.iterrows():
                            log_msg = f"{row['OpName']}: {row['Description']}"
                            log_messages.append(log_msg)
                        logs.extend(log_messages)
                except Exception as e:
                    print(f"⚠️ event.csv okuma hatası: {str(e)}")
            
            # trace.csv dosyasından da veri al
            trace_file = scenario_dir / "trace.csv"
            if trace_file.exists() and len(logs) < 500:  # Yeterli veri yoksa trace'den de al
                try:
                    df = pd.read_csv(trace_file, nrows=500)
                    if 'Title' in df.columns:
                        for _, row in df.iterrows():
                            log_msg = f"Task: {row['Title']} NumReports: {row['NumReports']}"
                            logs.append(log_msg)
                except Exception as e:
                    print(f"⚠️ trace.csv okuma hatası: {str(e)}")
                    
        except Exception as e:
            print(f"⚠️ {scenario_path} yüklenirken hata: {str(e)}")
        
        return logs
    
    def process_mega_dataset(self, data_dir="data/HDFS_big.log"):
        """Mega veri seti işleme ana fonksiyonu"""
        print("🚀 MEGA HDFS TRACE VERİ SETİ İŞLEME BAŞLADI")
        print("=" * 70)
        
        start_time = time.time()
        
        # 1. Tracebench verilerini keşfet
        anomaly_categories = self.explore_tracebench_data(data_dir)
        
        # 2. Çeşitli senaryoları örnekle
        selected_scenarios = self.sample_diverse_scenarios(anomaly_categories, samples_per_category=3)
        
        # 3. Seçilen senaryolardan log verilerini topla
        print("\n📚 Seçilen senaryolardan log verileri toplanıyor...")
        
        all_logs = []
        scenario_labels = []
        
        tracebench_path = Path(data_dir) / "tracebench"
        
        for category, scenarios in selected_scenarios.items():
            for scenario in scenarios:
                scenario_path = tracebench_path / scenario
                
                print(f"  {scenario} işleniyor...")
                scenario_logs = self.load_scenario_logs(scenario_path)
                
                if scenario_logs:
                    # Max 1000 log per scenario
                    sample_size = min(1000, len(scenario_logs))
                    sampled_logs = np.random.choice(scenario_logs, sample_size, replace=False)
                    
                    all_logs.extend(sampled_logs)
                    scenario_labels.extend([category] * len(sampled_logs))
                    
                    print(f"    ✅ {len(sampled_logs)} log eklendi")
        
        print(f"\n📊 Toplam toplanan log: {len(all_logs):,}")
        print(f"📊 Kategoriler: {len(set(scenario_labels))}")
        
        # 4. LogBERT ile anomali tespiti
        if len(all_logs) > 0:
            results = self.run_logbert_on_mega_data(all_logs, scenario_labels)
        else:
            print("❌ Hiç log verisi toplanamadı!")
            return None
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\n⏱️ TOPLAM İŞLEM SÜRESİ: {processing_time:.1f} saniye")
        print("=" * 70)
        print("✅ MEGA VERİ SETİ İŞLEME TAMAMLANDI!")
        
        return results
    
    def run_logbert_on_mega_data(self, logs, labels):
        """Toplanan mega veri üzerinde LogBERT çalıştır"""
        print("\n🧠 MEGA VERİ ÜZERİNDE LogBERT ANOMALİ TESPİTİ")
        print("=" * 50)
        
        # Veri ön işleme
        processed_logs = self.preprocess_mega_logs(logs)
        
        # BERT embeddings çıkarma
        embeddings = self.extract_embeddings_mega(processed_logs)
        
        # Anomali tespiti
        anomaly_results = self.detect_anomalies_mega(embeddings, labels)
        
        # Sonuçları analiz et ve görselleştir
        self.analyze_mega_results(anomaly_results, labels, processed_logs)
        
        return {
            'logs': processed_logs,
            'embeddings': embeddings,
            'anomaly_results': anomaly_results,
            'labels': labels
        }
    
    def preprocess_mega_logs(self, logs):
        """Mega veri log ön işleme"""
        print("📝 Mega veri log ön işleme...")
        
        import re
        processed = []
        
        for log in tqdm(logs[:self.sample_size], desc="Log preprocessing"):
            if isinstance(log, str):
                # Temel temizleme
                log = re.sub(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}', '[TIMESTAMP]', log)
                log = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d+)?', '[IP]', log)
                log = re.sub(r'blk_-?\d+', '[BLOCK_ID]', log)
                log = re.sub(r'\b\d{5,}\b', '[NUMBER]', log)
                
                processed.append(log.strip())
        
        print(f"✅ {len(processed)} log işlendi")
        return processed
    
    def extract_embeddings_mega(self, logs):
        """Mega veri için BERT embeddings"""
        print("🧠 Mega veri BERT embeddings...")
        
        all_embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(logs), self.batch_size), desc="BERT embedding"):
                batch = logs[i:i+self.batch_size]
                
                try:
                    encoded = self.tokenizer(
                        batch,
                        truncation=True,
                        padding=True,
                        max_length=self.max_length,
                        return_tensors='pt'
                    )
                    
                    input_ids = encoded['input_ids'].to(self.device)
                    attention_mask = encoded['attention_mask'].to(self.device)
                    
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    
                    all_embeddings.append(embeddings)
                    
                    # Memory cleanup
                    del input_ids, attention_mask, outputs
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                except Exception as e:
                    continue
        
        final_embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])
        print(f"✅ {final_embeddings.shape[0]} embedding çıkarıldı")
        
        return final_embeddings
    
    def detect_anomalies_mega(self, embeddings, labels):
        """Mega veri anomali tespiti"""
        print("🔍 Mega veri anomali tespiti...")
        
        if len(embeddings) == 0:
            return None
        
        # Normalize
        normalized = self.scaler.fit_transform(embeddings)
        
        # MiniBatch K-Means clustering
        n_clusters = max(5, min(20, len(set(labels))))
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(normalized)
        
        # Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_scores = iso_forest.fit_predict(normalized)
        
        results = {
            'cluster_labels': cluster_labels,
            'anomaly_scores': anomaly_scores,
            'normalized_embeddings': normalized
        }
        
        anomaly_count = np.sum(anomaly_scores == -1)
        print(f"🚨 {anomaly_count} anomali tespit edildi ({anomaly_count/len(embeddings)*100:.1f}%)")
        
        return results
    
    def analyze_mega_results(self, results, labels, logs):
        """Mega veri sonuçlarını analiz et"""
        print("📊 Mega veri sonuçları analiz ediliyor...")
        
        if results is None:
            return
        
        # Kategori bazlı anomali analizi
        category_anomalies = {}
        
        for i, (label, anomaly_score) in enumerate(zip(labels, results['anomaly_scores'])):
            if label not in category_anomalies:
                category_anomalies[label] = {'total': 0, 'anomalies': 0}
            
            category_anomalies[label]['total'] += 1
            if anomaly_score == -1:
                category_anomalies[label]['anomalies'] += 1
        
        # Sonuçları yazdır
        print("\n📈 KATEGORİ BAZLI ANOMALİ ANALİZİ:")
        for category, stats in category_anomalies.items():
            total = stats['total']
            anomalies = stats['anomalies']
            percentage = (anomalies / total * 100) if total > 0 else 0
            print(f"  {category}: {anomalies}/{total} ({percentage:.1f}%)")
        
        # Görselleştirme
        self.visualize_mega_results(results, labels)
        
        # Sonuçları kaydet
        self.save_mega_results(results, labels, logs, category_anomalies)
    
    def visualize_mega_results(self, results, labels):
        """Mega veri sonuçlarını görselleştir"""
        print("📊 Mega veri görselleştirmesi...")
        
        embeddings = results['normalized_embeddings']
        anomaly_scores = results['anomaly_scores']
        
        # PCA ile boyut indirgeme
        pca = IncrementalPCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        
        plt.figure(figsize=(15, 10))
        
        # Anomali görselleştirmesi
        plt.subplot(2, 2, 1)
        colors = ['blue' if score != -1 else 'red' for score in anomaly_scores]
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, alpha=0.6, s=20)
        plt.title('Mega Veri Anomali Tespiti')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        
        # Kategori dağılımı
        plt.subplot(2, 2, 2)
        label_counts = pd.Series(labels).value_counts()
        plt.bar(range(len(label_counts)), label_counts.values)
        plt.title('Kategori Dağılımı')
        plt.xticks(range(len(label_counts)), label_counts.index, rotation=45)
        
        # Anomali oranları
        plt.subplot(2, 2, 3)
        anomaly_by_category = {}
        for i, (label, score) in enumerate(zip(labels, anomaly_scores)):
            if label not in anomaly_by_category:
                anomaly_by_category[label] = []
            anomaly_by_category[label].append(1 if score == -1 else 0)
        
        categories = list(anomaly_by_category.keys())
        anomaly_rates = [np.mean(anomaly_by_category[cat]) * 100 for cat in categories]
        
        plt.bar(range(len(categories)), anomaly_rates)
        plt.title('Kategoriye Göre Anomali Oranları (%)')
        plt.xticks(range(len(categories)), categories, rotation=45)
        plt.ylabel('Anomali Oranı (%)')
        
        # Cluster dağılımı
        plt.subplot(2, 2, 4)
        cluster_labels = results['cluster_labels']
        unique_clusters = np.unique(cluster_labels)
        for cluster in unique_clusters:
            mask = cluster_labels == cluster
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                       label=f'Cluster {cluster}', alpha=0.6, s=20)
        plt.title('Cluster Dağılımı')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('results/mega_data_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Mega veri görselleştirmesi kaydedildi: results/mega_data_analysis.png")
    
    def save_mega_results(self, results, labels, logs, category_stats):
        """Mega veri sonuçlarını kaydet"""
        print("💾 Mega veri sonuçları kaydediliyor...")
        
        # Ana raporu kaydet
        with open('results/mega_data_report.txt', 'w', encoding='utf-8') as f:
            f.write("🚀 MEGA HDFS TRACE VERİ SETİ ANALİZ RAPORU\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Toplam işlenen log: {len(logs):,}\n")
            f.write(f"Toplam kategoriler: {len(set(labels))}\n")
            f.write(f"Tespit edilen anomaliler: {np.sum(results['anomaly_scores'] == -1):,}\n")
            f.write(f"Anomali oranı: {np.sum(results['anomaly_scores'] == -1) / len(results['anomaly_scores']) * 100:.2f}%\n\n")
            
            f.write("📊 KATEGORİ BAZLI İSTATİSTİKLER:\n")
            f.write("-" * 40 + "\n")
            for category, stats in category_stats.items():
                percentage = (stats['anomalies'] / stats['total'] * 100) if stats['total'] > 0 else 0
                f.write(f"{category}: {stats['anomalies']}/{stats['total']} ({percentage:.1f}%)\n")
        
        print("✅ Mega veri raporu kaydedildi: results/mega_data_report.txt")

def main():
    """Ana fonksiyon"""
    print("🌟 MEGA HDFS TRACE VERİ ANALİZİ BAŞLADI")
    print("=" * 70)
    
    # Mega LogBERT sistemi oluştur
    mega_logbert = MegaDataLogBERT(
        max_length=64,    # Daha küçük max length
        batch_size=4      # Daha küçük batch size
    )
    
    # Mega veri setini işle
    results = mega_logbert.process_mega_dataset()
    
    return results

if __name__ == "__main__":
    results = main()
