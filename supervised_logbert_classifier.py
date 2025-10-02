import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, AdamW
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import pickle
from pathlib import Path
warnings.filterwarnings('ignore')

class SupervisedLogBERT(nn.Module):
    """Etiketli anomali sınıflandırması için LogBERT"""
    
    def __init__(self, model_name='bert-base-uncased', num_classes=11, hidden_size=768, dropout_rate=0.3):
        super(SupervisedLogBERT, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits

class LogDataset(Dataset):
    """Log verileri için PyTorch Dataset"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class SupervisedHDFSClassifier:
    """Supervised HDFS Log Anomali Sınıflandırıcısı"""
    
    def __init__(self, model_name='bert-base-uncased', max_length=128, batch_size=16):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Kullanılan cihaz: {self.device}")
        
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.batch_size = batch_size
        
        # Sınıf etiketleri
        self.class_names = [
            'Normal',
            'Data_corrupt', 
            'Data_cut',
            'Data_loss',
            'Net_disconnect',
            'Net_slow', 
            'Proc_kill',
            'Proc_suspend',
            'Sys_dead',
            'Sys_panic',
            'Complex'
        ]
        
        self.label_to_id = {label: idx for idx, label in enumerate(self.class_names)}
        self.id_to_label = {idx: label for idx, label in enumerate(self.class_names)}
        
        self.model = SupervisedLogBERT(num_classes=len(self.class_names))
        self.model.to(self.device)
    
    def load_labeled_data_from_scenarios(self, data_dir="data/HDFS_big.log"):
        """Senaryolardan etiketli veri yükle"""
        print("📊 ETİKETLİ VERİ SETİ OLUŞTURULUYOR")
        print("=" * 50)
        
        tracebench_path = Path(data_dir) / "tracebench"
        
        all_texts = []
        all_labels = []
        
        # Her kategoriden veri topla
        category_counts = {}
        
        for folder in tqdm(tracebench_path.iterdir(), desc="Klasörler taranıyor"):
            if folder.is_dir():
                folder_name = folder.name
                
                # Kategori belirle
                category = self._determine_category(folder_name)
                if category is None:
                    continue
                
                # Event.csv dosyasından logları yükle
                event_file = folder / "event.csv"
                if event_file.exists():
                    try:
                        df = pd.read_csv(event_file, nrows=200)  # Her senaryodan 200 örnek
                        if 'Description' in df.columns and 'OpName' in df.columns:
                            for _, row in df.iterrows():
                                log_text = f"{row['OpName']}: {row['Description']}"
                                all_texts.append(log_text)
                                all_labels.append(category)
                                
                                # Sayaç
                                if category not in category_counts:
                                    category_counts[category] = 0
                                category_counts[category] += 1
                    except Exception as e:
                        continue
        
        print("\n📈 TOPLANAN ETİKETLİ VERİ:")
        for category, count in category_counts.items():
            print(f"  {category}: {count:,} örnek")
        
        print(f"\nToplam etiketli örnek: {len(all_texts):,}")
        
        return all_texts, all_labels
    
    def _determine_category(self, folder_name):
        """Klasör ismine göre kategori belirle"""
        if folder_name.startswith('NM_'):
            return 'Normal'
        elif 'corruptBlk' in folder_name or 'corruptMeta' in folder_name:
            return 'Data_corrupt'
        elif 'cutBlk' in folder_name or 'cutMeta' in folder_name:
            return 'Data_cut'
        elif 'lossBlk' in folder_name or 'lossMeta' in folder_name:
            return 'Data_loss'
        elif 'disconnectDN' in folder_name:
            return 'Net_disconnect'
        elif 'slowDN' in folder_name or 'slowHDFS' in folder_name:
            return 'Net_slow'
        elif 'killDN' in folder_name:
            return 'Proc_kill'
        elif 'suspendDN' in folder_name:
            return 'Proc_suspend'
        elif 'deadDN' in folder_name:
            return 'Sys_dead'
        elif 'panicDN' in folder_name:
            return 'Sys_panic'
        elif folder_name.startswith('COM_'):
            return 'Complex'
        else:
            return None
    
    def prepare_data(self, texts, labels):
        """Veriyi eğitim için hazırla"""
        print("\n🔧 VERİ EĞİTİM İÇİN HAZIRLANIYOR...")
        
        # Etiketleri sayısal değerlere çevir
        numeric_labels = [self.label_to_id[label] for label in labels]
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            texts, numeric_labels, 
            test_size=0.2, 
            random_state=42, 
            stratify=numeric_labels
        )
        
        print(f"Eğitim seti: {len(X_train):,} örnek")
        print(f"Test seti: {len(X_test):,} örnek")
        
        # Dataset oluştur
        train_dataset = LogDataset(X_train, y_train, self.tokenizer, self.max_length)
        test_dataset = LogDataset(X_test, y_test, self.tokenizer, self.max_length)
        
        # DataLoader oluştur
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, test_loader, X_test, y_test
    
    def train_model(self, train_loader, test_loader, epochs=3, learning_rate=2e-5):
        """Modeli eğit"""
        print(f"\n🚀 MODEL EĞİTİMİ BAŞLADI ({epochs} epoch)")
        print("=" * 50)
        
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        for epoch in range(epochs):
            print(f"\n📚 Epoch {epoch + 1}/{epochs}")
            
            # Training
            self.model.train()
            total_loss = 0
            correct_predictions = 0
            total_predictions = 0
            
            progress_bar = tqdm(train_loader, desc="Eğitim")
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                predictions = torch.argmax(outputs, dim=-1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)
                
                # Progress bar güncelle
                accuracy = correct_predictions / total_predictions * 100
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{accuracy:.1f}%'
                })
            
            # Epoch sonuçları
            epoch_loss = total_loss / len(train_loader)
            epoch_accuracy = correct_predictions / total_predictions * 100
            
            print(f"Epoch {epoch + 1} Sonuçları:")
            print(f"  Ortalama Loss: {epoch_loss:.4f}")
            print(f"  Eğitim Doğruluğu: {epoch_accuracy:.2f}%")
            
            # Validation
            val_accuracy = self.evaluate_model(test_loader, verbose=False)
            print(f"  Validasyon Doğruluğu: {val_accuracy:.2f}%")
        
        print("\n✅ MODEL EĞİTİMİ TAMAMLANDI!")
    
    def evaluate_model(self, test_loader, verbose=True):
        """Modeli değerlendir"""
        if verbose:
            print("\n📊 MODEL DEĞERLENDİRİLİYOR...")
        
        self.model.eval()
        correct_predictions = 0
        total_predictions = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                predictions = torch.argmax(outputs, dim=-1)
                
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = correct_predictions / total_predictions * 100
        
        if verbose:
            print(f"✅ Test Doğruluğu: {accuracy:.2f}%")
            
            # Sınıflandırma raporu
            class_names = [self.id_to_label[i] for i in range(len(self.class_names))]
            report = classification_report(all_labels, all_predictions, target_names=class_names)
            print(f"\n📋 SINIFLANDIRMA RAPORU:\n{report}")
            
            # Confusion matrix
            self.plot_confusion_matrix(all_labels, all_predictions)
        
        return accuracy
    
    def plot_confusion_matrix(self, true_labels, pred_labels):
        """Confusion matrix çiz"""
        cm = confusion_matrix(true_labels, pred_labels)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title('Confusion Matrix - Supervised LogBERT')
        plt.xlabel('Tahmin Edilen')
        plt.ylabel('Gerçek')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('results/supervised_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Confusion matrix kaydedildi: results/supervised_confusion_matrix.png")
    
    def predict_single_log(self, text):
        """Tek bir log için tahmin yap"""
        self.model.eval()
        
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
        
        return predicted_class, confidence
    
    def save_model(self, path='models/supervised_logbert.pth'):
        """Modeli kaydet"""
        Path(path).parent.mkdir(exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'class_names': self.class_names,
            'label_to_id': self.label_to_id,
            'id_to_label': self.id_to_label
        }, path)
        print(f"✅ Model kaydedildi: {path}")
    
    def run_supervised_training(self):
        """Tam supervised eğitim süreci"""
        print("🎯 SUPERVISED HDFS LOG SINIFLANDIRICI")
        print("=" * 60)
        
        # 1. Etiketli veri yükle
        texts, labels = self.load_labeled_data_from_scenarios()
        
        if len(texts) < 100:
            print("❌ Yeterli veri bulunamadı!")
            return None
        
        # 2. Veriyi hazırla
        train_loader, test_loader, X_test, y_test = self.prepare_data(texts, labels)
        
        # 3. Modeli eğit
        self.train_model(train_loader, test_loader, epochs=3)
        
        # 4. Modeli değerlendir
        self.evaluate_model(test_loader)
        
        # 5. Modeli kaydet
        self.save_model()
        
        # 6. Örnek tahminler
        print("\n🧪 ÖRNEK TAHMİNLER:")
        test_examples = [
            "getFileInfo: Success: return file status",
            "BLOCK* ask datanode to delete corrupted block", 
            "RPC:getFileInfo connection timeout",
            "PacketResponder terminating due to network error"
        ]
        
        for example in test_examples:
            predicted_class, confidence = self.predict_single_log(example)
            print(f"Log: {example[:50]}...")
            print(f"Tahmin: {predicted_class} (Güven: {confidence:.3f})")
            print("-" * 50)
        
        print("\n✅ SUPERVISED SİSTEM HAZIR!")
        return self.model

def main():
    """Ana fonksiyon"""
    classifier = SupervisedHDFSClassifier(batch_size=8)  # Küçük batch size
    model = classifier.run_supervised_training()
    return classifier

if __name__ == "__main__":
    classifier = main()
