import torch
from transformers import BertTokenizer
from supervised_logbert_classifier import SupervisedLogBERT
import pickle
from pathlib import Path

class SupervisedLogPredictor:
    """Eğitilmiş supervised modelle gerçek zamanlı tahmin"""
    
    def __init__(self, model_path='models/supervised_logbert.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Cihaz: {self.device}")
        
        # Model yükle
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.class_names = checkpoint['class_names']
        self.label_to_id = checkpoint['label_to_id']
        self.id_to_label = checkpoint['id_to_label']
        
        # Model oluştur ve yükle
        self.model = SupervisedLogBERT(num_classes=len(self.class_names))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = 128
        
        print("✅ Supervised model yüklendi!")
        print(f"📊 Sınıflar: {self.class_names}")
    
    def predict_with_confidence(self, text):
        """Tek log için tahmin + güven skoru"""
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probabilities = torch.softmax(outputs, dim=-1)
            prediction = torch.argmax(outputs, dim=-1)
        
        predicted_class = self.id_to_label[prediction.item()]
        confidence = probabilities[0][prediction].item()
        
        # Tüm sınıf olasılıkları
        all_probs = {}
        for i, class_name in enumerate(self.class_names):
            all_probs[class_name] = probabilities[0][i].item()
        
        return predicted_class, confidence, all_probs
    
    def analyze_log_batch(self, log_list):
        """Log listesi için toplu analiz"""
        results = []
        
        print(f"\n🔍 {len(log_list)} log analiz ediliyor...")
        
        for i, log in enumerate(log_list):
            predicted_class, confidence, all_probs = self.predict_with_confidence(log)
            
            # Anomali mi?
            is_anomaly = predicted_class != 'Normal'
            
            result = {
                'log_id': i + 1,
                'log_text': log[:100] + "..." if len(log) > 100 else log,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'is_anomaly': is_anomaly,
                'all_probabilities': all_probs
            }
            
            results.append(result)
        
        return results
    
    def print_analysis_results(self, results):
        """Analiz sonuçlarını yazdır"""
        print("\n📊 SUPERVISED TAHMİN SONUÇLARI:")
        print("=" * 80)
        
        anomaly_count = sum(1 for r in results if r['is_anomaly'])
        normal_count = len(results) - anomaly_count
        
        print(f"Toplam Log: {len(results)}")
        print(f"Anomali: {anomaly_count} ({anomaly_count/len(results)*100:.1f}%)")
        print(f"Normal: {normal_count} ({normal_count/len(results)*100:.1f}%)")
        
        print(f"\n{'ID':<3} {'Sınıf':<15} {'Güven':<8} {'Durum':<8} Log")
        print("-" * 80)
        
        for result in results:
            status = "🚨 ANO" if result['is_anomaly'] else "✅ NOR"
            print(f"{result['log_id']:<3} {result['predicted_class']:<15} "
                  f"{result['confidence']:.3f}    {status:<8} {result['log_text']}")
        
        # Anomali türü dağılımı
        if anomaly_count > 0:
            print(f"\n🚨 ANOMALİ TÜRÜ DAĞILIMI:")
            anomaly_types = {}
            for result in results:
                if result['is_anomaly']:
                    atype = result['predicted_class']
                    anomaly_types[atype] = anomaly_types.get(atype, 0) + 1
            
            for atype, count in sorted(anomaly_types.items(), key=lambda x: x[1], reverse=True):
                percentage = count / anomaly_count * 100
                print(f"  {atype}: {count} ({percentage:.1f}%)")

def test_supervised_predictor():
    """Supervised tahmin sistemini test et"""
    print("🎯 SUPERVISED LOG TAHMİN SİSTEMİ TEST")
    print("=" * 60)
    
    # Model yükle
    predictor = SupervisedLogPredictor()
    
    # Test logları
    test_logs = [
        "getFileInfo: Success: return file status for /user/data/file1.txt",
        "BLOCK* ask datanode to delete corrupted block blk_1234567890",
        "RPC connection timeout for getFileInfo operation after 30 seconds",
        "PacketResponder terminating due to network disconnection",
        "DataNode process killed by system administrator",
        "System panic: kernel memory corruption detected",
        "HDFS operation suspended due to resource constraints", 
        "Block metadata file corrupted, unable to read block info",
        "Creating new directory /user/data/logs successfully",
        "Data loss detected in block blk_9876543210",
        "Network slowdown detected, reducing transfer rate",
        "Complex distributed system failure involving multiple components"
    ]
    
    # Analiz yap
    results = predictor.analyze_log_batch(test_logs)
    
    # Sonuçları göster
    predictor.print_analysis_results(results)
    
    # Detaylı örnek
    print(f"\n🔍 DETAYLI ÖRNEK ANALİZ:")
    print("-" * 50)
    example_log = "BLOCK* datanode corrupted metadata detected in block verification"
    predicted_class, confidence, all_probs = predictor.predict_with_confidence(example_log)
    
    print(f"Log: {example_log}")
    print(f"Ana Tahmin: {predicted_class} (Güven: {confidence:.3f})")
    print(f"\nTüm Sınıf Olasılıkları:")
    
    sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
    for class_name, prob in sorted_probs[:5]:  # Top 5
        print(f"  {class_name}: {prob:.3f}")
    
    return results

if __name__ == "__main__":
    results = test_supervised_predictor()
