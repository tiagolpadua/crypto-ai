import numpy as np
import pandas as pd

y_test_inverse = np.array([[31925.105], [31921.627], [31918.06 ]], np.float32)

y_hat_inverse = np.array([[5.105], [6.627], [7.06 ]], np.float32)

# print(y_test_inverse.tolist())

# print(y_test_inverse[0:, 0])

# print(y_hat_inverse)

df = pd.DataFrame({'y_test_inverse': y_test_inverse[0:, 0], 'y_hat_inverse': y_hat_inverse[0:, 0]})

print(df)

df.to_excel('exemplo.xlsx', index=False)

df.to_csv('exemplo.csv', index=False)