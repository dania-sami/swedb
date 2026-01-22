import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("yield_curve_data.csv")

plt.plot(data["maturity_years"], data["base_yield"], marker="o", label="Base")
plt.plot(data["maturity_years"], data["stress_yield"], marker="o", label="Stress")
plt.xlabel("Maturity in years")
plt.ylabel("Yield in percent")
plt.title("Illustrative Yield Curves")
plt.legend()
plt.tight_layout()
plt.show()
