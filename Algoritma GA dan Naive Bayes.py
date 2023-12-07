import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import accuracy_score

class NB:
    def prediksi(self, X):
        prediksi = [self._prediksi(i) for i in X]
        return np.array(prediksi)

    def _prediksi(self, x):
        posteriors = []
        for idx, nama_kelas in enumerate(self.kelas):
            prior = np.log(self.prior[idx])
            posterior = sum(np.log(self.fungsi_derivatif(idx, x)))
            posterior = prior + posterior
            posteriors.append(posterior)

        return self.kelas[np.argmax(posteriors)]

    def fungsi_derivatif(self, kelas_indeks, x):
        mean = self.mean[kelas_indeks]
        var = self.var[kelas_indeks]
        pembilang = np.exp(-(x - mean) ** 2 / (2 * var))
        penyebut = np.sqrt(2 * np.pi * var)
        return pembilang / penyebut

    def fit(self, X, Y):
        n_baris, n_kolom = X.shape
        self.kelas = np.unique(Y)
        jumlah_kelas = len(self.kelas)

        self.mean = np.zeros((jumlah_kelas, n_kolom), dtype=np.float64)
        self.var = np.zeros((jumlah_kelas, n_kolom), dtype=np.float64)
        self.prior = np.zeros(jumlah_kelas, dtype=np.float64)

        for idx, nama_kelas in enumerate(self.kelas):
            X_kelas = X[Y == nama_kelas]
            self.mean[idx, :] = X_kelas.mean(axis=0)
            self.var[idx, :] = X_kelas.var(axis=0)
            self.prior[idx] = X_kelas.shape[0] / n_baris


class FeatureSelectionGA:
    def __init__(self, model, feature_names, population_size=100, generations=50, crossover_prob=0.8, mutation_prob=0.2):
        self.model = model
        self.feature_names = feature_names
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob

    def _fitness(self, X_train, X_test, y_train, y_test, selected_features):
        model = self.model
        model.fit(X_train[:, selected_features], y_train)
        y_pred = model.prediksi(X_test[:, selected_features])
        return accuracy_score(y_test, y_pred)

    def _initialize_population(self, num_features):
        return np.random.choice([0, 1], size=(self.population_size, num_features))

    def _crossover(self, parents):
        children = []
        for i in range(0, len(parents), 2):
            parent1, parent2 = parents[i], parents[i + 1]
            if np.random.rand() < self.crossover_prob:
                crossover_point = np.random.randint(1, len(parent1) - 1)
                child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
                child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            children.extend([child1, child2])
        return np.array(children)

    def _mutation(self, children):
        mutation_mask = (np.random.rand(*children.shape) < self.mutation_prob).astype(int)
        return (children + mutation_mask) % 2

    def _select_features(self, X_train, X_test, y_train, y_test, population):
        fitness_values = []
        for features in population:
            fitness_values.append(self._fitness(X_train, X_test, y_train, y_test, np.where(features == 1)[0]))
        return np.array(fitness_values)

    def optimize(self, X_train, X_test, y_train, y_test):
        num_features = X_train.shape[1]
        population = self._initialize_population(num_features)

        for generation in tqdm(range(self.generations)):
            fitness_values = self._select_features(X_train, X_test, y_train, y_test, population)
            parents = population[np.argsort(fitness_values)[::-1][:self.population_size // 2]]

            children = self._crossover(parents)
            children = self._mutation(children)

            population = np.vstack([parents, children])

        best_features = np.where(population[0] == 1)[0]
        selected_feature_names = [self.feature_names[i] for i in best_features]
        return best_features, selected_feature_names


# Load your dataset
df = pd.read_csv('breast_cancer.csv')
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

X = df.iloc[:, 2:].values
y = df.iloc[:, 1].values
feature_names = df.columns[2:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Naive Bayes model
nb_model = NB()

# Use genetic algorithm for feature selection
feature_selector = FeatureSelectionGA(model=nb_model, feature_names=feature_names)
best_features, selected_feature_names = feature_selector.optimize(X_train, X_test, y_train, y_test)

print("\n")
print("Best Features:", best_features)
print("Selected Feature Names:", selected_feature_names)
print("\n")

# Train Naive Bayes with the best features
nb_model.fit(X_train[:, best_features], y_train)
y_pred = nb_model.prediksi(X_test[:, best_features])
accuracy = accuracy_score(y_test, y_pred)
print(f"With feature selection, accuracy: {accuracy * 100:.2f}%")

# Train Naive Bayes without feature selection
nb_model_no_fs = NB()
nb_model_no_fs.fit(X_train, y_train)
y_pred_no_fs = nb_model_no_fs.prediksi(X_test)
accuracy_no_fs = accuracy_score(y_test, y_pred_no_fs)
print(f"Without feature selection, accuracy: {accuracy_no_fs * 100:.2f}%")
