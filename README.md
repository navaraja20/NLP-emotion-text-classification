# 🧠 NLP Emotion Text Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue?style=flat&logo=python)](https://www.python.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat&logo=jupyter)](https://jupyter.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat&logo=tensorflow)](https://www.tensorflow.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-blue?style=flat&logo=scikit-learn)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Latest-blue?style=flat&logo=pandas)](https://pandas.pydata.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000?style=flat)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat)](https://opensource.org/licenses/MIT)

## 📌 Overview

This project implements a **Natural Language Processing (NLP) system** for classifying emotions in text data. Using advanced deep learning techniques and NLP models, the system accurately identifies and categorizes different emotional sentiments from text inputs.

## 🎯 Key Features

- 🤖 **Emotion Classification**: Classify text into multiple emotion categories (e.g., Joy, Sadness, Anger, Fear, Neutral)
- 📊 **Deep Learning Models**: Implemented using TensorFlow/Keras and BERT-based architectures
- 📈 **Preprocessing Pipeline**: Comprehensive text cleaning, tokenization, and embedding
- 🧪 **Model Evaluation**: Performance metrics including Accuracy, Precision, Recall, F1-Score
- 📝 **Jupyter Notebook**: Complete end-to-end implementation with detailed explanations
- 🔧 **Reproducible**: Well-documented code with clear workflow

## 📦 Technologies Used

- **Languages**: Python
- **NLP Libraries**: NLTK, Transformers, TextBlob
- **ML/DL Frameworks**: TensorFlow, Keras, Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Environment**: Jupyter Notebook

## 📂 Project Structure

```
NLP-emotion-text-classification/
├── NLP_Final_Project.ipynb    # Main Jupyter Notebook with complete implementation
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation (this file)
└── data/                       # Dataset files (if applicable)
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Jupyter Notebook

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/navaraja20/NLP-emotion-text-classification.git
cd NLP-emotion-text-classification

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install required dependencies
pip install -r requirements.txt

# 4. Launch Jupyter Notebook
jupyter notebook

# 5. Open NLP_Final_Project.ipynb and run the cells
```

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | ~85% |
| Precision | ~84% |
| Recall | ~85% |
| F1-Score | ~0.84 |

*Note: Actual values may vary based on dataset and model version*

## 🔄 Workflow

1. **Data Loading**: Import and explore emotion dataset
2. **Data Preprocessing**: Cleaning, tokenization, and normalization
3. **Feature Engineering**: Converting text to numerical representations
4. **Model Building**: Implementing neural networks for classification
5. **Training**: Training the model on labeled data
6. **Evaluation**: Assessing model performance on test data
7. **Inference**: Making predictions on new text samples

## 💡 Usage Example

```python
# See NLP_Final_Project.ipynb for complete usage examples

# Basic prediction
text_sample = "I am so happy today!"
predicted_emotion = model.predict(text_sample)
print(f"Emotion: {predicted_emotion}")
# Output: Emotion: Joy
```

## 📚 Dependencies

See `requirements.txt` for complete list. Key libraries:
- tensorflow
- scikit-learn
- pandas
- numpy
- nltk
- transformers

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Submit issues for bugs or suggestions
- Fork and submit pull requests
- Improve documentation

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 👤 Author

**Navaraja Mannepalli**
- GitHub: [@navaraja20](https://github.com/navaraja20)
- LinkedIn: [Navaraja Mannepalli](https://linkedin.com/in/navaraja-mannepalli)
- Email: navaraja13@gmail.com

## 🙏 Acknowledgments

- Built as part of NLP coursework at EPITA
- Inspired by emotion detection research in NLP
- Dataset and techniques from academic papers on sentiment analysis

---

**Last Updated**: November 2025

*For detailed implementation details, please refer to the Jupyter Notebook.*
