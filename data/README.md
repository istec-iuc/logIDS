# 📊 HDFS Dataset

## 🗂️ Dataset Structure

### Original Dataset
- **Source**: HDFS Big Dataset (Kaggle)
- **Total Size**: 590MB
- **Format**: Raw HDFS log traces
- **Entries**: 30,000+ log messages

### Preprocessed Data Files

#### `normal_trace.csv` (1.9GB)
- **Content**: Normal HDFS operations
- **Records**: ~25,000 entries
- **Columns**: timestamp, operation, details, processed_text
- **Note**: Excluded from Git due to size (>100MB GitHub limit)

#### `failure_trace.csv` (246MB)  
- **Content**: Anomalous HDFS operations
- **Records**: ~5,000 entries
- **Anomaly Types**: 11 categories
- **Note**: Excluded from Git due to size (>100MB GitHub limit)

## 📥 Dataset Access

### Option 1: Download Original Dataset
```bash
# Download from Kaggle
kaggle datasets download -d uciml/hdfs-big-dataset
```

### Option 2: Generate Preprocessed Data
```bash
# Run the unsupervised analyzer to generate CSV files
python mega_logbert_analyzer.py
```

### Option 3: Use Sample Data
Small sample files are included for testing:
- `sample_normal.csv` (50 entries)
- `sample_failure.csv` (50 entries)

## 📊 Data Statistics

| Category | Count | Percentage |
|----------|-------|------------|
| **Normal Operations** | 25,372 | 85.5% |
| **Data_corrupt** | 1,256 | 4.2% |
| **Net_slow** | 1,089 | 3.7% |
| **Proc_kill** | 987 | 3.3% |
| **Data_loss** | 876 | 2.9% |
| **Other Anomalies** | 420 | 1.4% |

## 🔍 Data Format Example

```csv
timestamp,operation,details,processed_text,label
1547759200,READ,blk_123456789,block read operation completed,Normal
1547759300,CORRUPT,blk_987654321,data corruption detected in block,Data_corrupt
```

## ⚠️ Important Notes

1. **Large Files**: Main CSV files excluded from Git repository
2. **Privacy**: IP addresses masked as xxx.xxx.xxx.xxx
3. **Preprocessing**: Applied tokenization and normalization
4. **Labels**: Manually verified anomaly classifications

---
**To use the full dataset, please download or generate the preprocessed files as described above.**