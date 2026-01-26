# 🏦 Bank Customer Churn Prediction (DLU Course c.3)

Цей проєкт створено в рамках курсу DLU (Data Literacy University) для участі в змаганні Kaggle:  
**[Bank Customer Churn Prediction (DLU Course c.3)](https://www.kaggle.com/competitions/bank-customer-churn-prediction-dlu-course-c3)**

---

## 📊 Мета

Побудувати модель, яка передбачає ймовірність того, що клієнт банку припинить користуватися його послугами (churn / `Exited = 1`) на основі історичних даних.

---

## 🛠 Використані методи та Технології

- 📦 **Pandas, Scikit-Learn** — для обробки даних і моделювання
- 🔍 **Stratified train/validation split** (80/20) — для збереження балансу класів
- ⚙️ **Передобробка даних (`ColumnTransformer` + `Pipeline`):**
  - **Числові ознаки** (`CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`):
    - Імпутація пропусків середнім значенням (`SimpleImputer(strategy='mean')`)
    - Масштабування (`StandardScaler`)
  - **Категоріальні ознаки** (`Geography`, `Gender`):
    - Імпутація пропусків модою (`SimpleImputer(strategy='most_frequent')`)
    - Кодування (`OneHotEncoder(handle_unknown='ignore')`)
  - **Видалення нерелевантних колонок**: `id`, `Surname`, `CustomerId`
- 🤖 **Модель:** `LogisticRegression(max_iter=1000, random_state=42)`
- 🎯 **Метрики оцінки:**
  - `Accuracy`
  - `F1 Score`
  - `ROC AUC`
  - `Confusion Matrix`

---

## 🔁 Повний пайплайн реалізації

1. **Завантаження та аналіз даних**: Зчитування `train.csv`, `test.csv`.
2. **Розбиття даних**: Використання `train_test_split` з параметром `stratify=y` для коректного розподілу цільової змінної.
3. **Побудова препроцесора**: Створення пайплайнів для автоматичної обробки числових та категоріальних даних.
4. **Збереження проміжних даних**: Оброблені дані зберігаються у форматі `.parquet` для швидкого доступу.
5. **Навчання моделі**: Тренування логістичної регресії на підготовлених даних.
6. **Оцінка якості**: Розрахунок метрик на валідаційній вибірці.
7. **Збереження артефактів**: Модель зберігається у файл `log_reg.joblib`.
8. **Функція передбачення**: Реалізовано функцію `predict_raw_df_preprocessor`, яка приймає сирі дані, обробляє їх і видає ймовірності.
9. **Сабміт**: Генерація файлу `submission_log_reg.csv` для Kaggle.

---

## 📁 Структура файлів

- `train.csv`, `test.csv`, `sample_submission.csv` — вхідні дані з Kaggle
- `submission_log_reg.csv` — згенерований файл з передбаченнями
- `log_reg.joblib` — збережена навчена модель
- `processed_data/` — папка з обробленими даними у форматі parquet
- `notebook.ipynb` — Jupyter Notebook з повним кодом рішення

---

## 🚀 Як запустити

1. Відкрити Google Colab або Jupyter Notebook.
2. Завантажити необхідні бібліотеки (`pandas`, `sklearn`, `joblib`, `matplotlib`).
3. Завантажити датасети у робочу директорію.
4. Виконати всі комірки ноутбука послідовно.
5. Отримати файл `submission_log_reg.csv` для завантаження на Kaggle.

---

## 🏁 Результати моделі

Логістична регресія показала наступні результати на валідаційній вибірці:

- **Train Accuracy**: ~0.8745
- **Val Accuracy**: ~0.8697
- **ROC AUC**: ~0.8797

Модель демонструє стабільність (відсутність сильного перенавчання) і значно перевершує випадкове вгадування (ROC AUC > 0.5).

---

## 📌 Приклад використання функції передбачення

```python
# Завантаження моделі
model = joblib.load('log_reg.joblib')

# Передбачення на нових сирих даних
probas = predict_raw_df_preprocessor(preprocessor, model, new_data_df)
```
