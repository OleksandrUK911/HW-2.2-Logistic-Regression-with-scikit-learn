# 🏦 Bank Customer Churn Prediction (DLU Course c.3)

Цей проєкт створено в рамках курсу DLU (Data Literacy University) для участі в змаганні Kaggle:  
**[Bank Customer Churn Prediction (DLU Course c.3)](https://www.kaggle.com/competitions/bank-customer-churn-prediction-dlu-course-c3)**

---

## 📊 Мета

Побудувати модель, яка передбачає ймовірність того, що клієнт банку припинить користуватися його послугами (churn / `Exited = 1`) на основі історичних даних.

---

## 🛠 Використані методи

- 📦 **Pandas, Scikit-Learn** — для обробки даних і моделювання
- 🔍 **Stratified train/validation split**
- ⚙️ **Передобробка:**
  - Масштабування числових ознак (`StandardScaler`)
  - Імпутація пропущених значень (`SimpleImputer`)
  - Кодування категоріальних ознак (`OneHotEncoder`)
  - Видалення нерелевантних колонок (`id`, `Surname`, `CustomerId`)
- 🤖 **Модель:** `LogisticRegression`
- 🎯 **Метрики оцінки:**
  - `Accuracy`
  - `F1 Score`
  - `ROC AUC`
  - `Confusion Matrix`
  - `ROC Curve`

---

## 🔁 Повний пайплайн

1. Зчитування `train.csv`, `test.csv`, `sample_submission.csv`
2. Розбиття на train/validation з використанням `train_test_split` + stratify
3. Побудова окремих датасетів `train_inputs`, `train_targets`, `val_inputs`, `val_targets`
4. Побудова препроцесора (`ColumnTransformer`)
5. Навчання моделі `LogisticRegression`
6. Оцінка на validation
7. Побудова submission-файлу `submission_log_reg.csv`
8. Збереження моделі (`joblib`)
9. Створення функції `predict_raw_df_preprocessor(...)` для передбачення на нових даних

---

## 📁 Файли

- `train.csv`, `test.csv`, `sample_submission.csv` — дані з Kaggle
- `submission_log_reg.csv` — файл, готовий для сабміту
- `log_reg.joblib` — збережена модель
- `notebook.ipynb` — основна реалізація (працює у Google Colab)

---

## 🚀 Як запустити

1. Відкрити Google Colab.
2. Завантажити CSV-файли в `/content/`.
3. Запустити cells у ноутбуці.
4. Згенерувати `submission_log_reg.csv`.
5. Завантажити на Kaggle через вкладку **Submit Predictions**.

---

## 📌 Приклад сабміту
CustomerId,Exited
15634602,0
15647311,0
15619304,0
15701354,0


---

## 🏁 Результат

Базова модель `LogisticRegression` із коректною передобробкою показала стабільні метрики (accuracy, ROC AUC) і перевершує наївну модель мажоритарного класу.
