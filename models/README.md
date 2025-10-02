# 🎯 HDFS LogBERT Trained Models

## 📋 Model Files

### `supervised_logbert.pth` (418MB)
- **Model Type**: Fine-tuned BERT-base-uncased
- **Architecture**: BERT + Dropout + Linear Classifier  
- **Classes**: 11 anomaly categories + normal
- **Accuracy**: 94.97%
- **Training Time**: 21+ hours
- **Input**: HDFS log text sequences
- **Output**: Anomaly classification probabilities

### 📥 Model Download
Due to GitHub file size limitations (100MB), the trained model is not included in this repository.

**To obtain the trained model:**
1. **Train from scratch**: Run `python supervised_logbert_classifier.py`
2. **Contact author**: Request pre-trained model
3. **Alternative**: Use smaller BERT variants

### 🏗️ Model Architecture
```python
SupervisedLogBERT(
  (bert): BertModel           # Pre-trained BERT-base-uncased
  (dropout): Dropout(0.1)     # Regularization layer
  (classifier): Linear(768 → 12)  # 11 anomaly classes + normal
)
```

### 📊 Model Performance
- **Parameters**: ~110M (BERT) + 9K (classifier) 
- **Inference Time**: ~95ms per log
- **Memory Usage**: ~2GB GPU / ~6GB RAM
- **Batch Processing**: Up to 64 logs simultaneously

### 🔧 Usage Example
```python
from supervised_predictor_test import LogPredictor

# Load trained model
predictor = LogPredictor()

# Predict anomaly
log_text = "Block corruption detected in blk_123456789" 
result = predictor.predict_log_type(log_text)

print(f"Anomaly: {result['class']} (Confidence: {result['confidence']:.2f})")
# Output: Anomaly: Data_corrupt (Confidence: 0.94)
```

---
**Note**: Model training requires HDFS dataset and significant computational resources.