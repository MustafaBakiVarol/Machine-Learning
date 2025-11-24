from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

# MNIST verisini indir
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

# Özellikler (piksel değerleri) ve etiketler
X, y = mnist["data"], mnist["target"]

# Verinin boyutuna bakalım
print(X.shape)  # (70000, 784)

# İlk görüntüyü al
some_digit = X[0]

# 28x28 formuna geri dönüştür
some_digit_image = some_digit.reshape(28, 28)

# Görselleştir
plt.imshow(some_digit_image, cmap='binary')
plt.axis('off')
plt.show()
