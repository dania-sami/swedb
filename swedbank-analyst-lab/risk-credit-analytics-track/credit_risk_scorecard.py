import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("credit_data.csv")

X = data[["income_k", "debt_k", "num_late_payments", "years_with_bank"]]
y = data["defaulted"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Classification report")
print(classification_report(y_test, y_pred))
