import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("customer_data.csv")

X = data[["age", "tenure_years", "products_owned", "avg_balance"]]
y = data["churned"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Classification report")
print(classification_report(y_test, y_pred))
