import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Function to load and display dataset
def load_data():
    return pd.read_csv('diabetes_psd.csv')

# Home page function
def Home():
    st.title("Klasifikasi Diabetes Menggunakan 3 Model")
    st.header("Tahapan Proses:")
    st.write("1. **Data Preprocessing**")
    st.write("2. **Pembagian Data**")
    st.write("3. **Evaluasi Model**")

# Preprocessing page function
def Preprocessing():
    st.title("Preprocessing Data")
    data = load_data()
    st.write("10 Data Awal:", data.head(10))

    # Filter hanya kolom numerik untuk deteksi outliers
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    z_score = np.abs((data[numeric_cols] - data[numeric_cols].mean()) / data[numeric_cols].std())
    threshold = 3
    outliers = (z_score > threshold)

    outlier_data = data[outliers.any(axis=1)]
    st.write("Data outliers:", outlier_data)

    outlier_per_column = outliers.sum(axis=0)
    st.write("Jumlah outliers per kolom:", outlier_per_column)

    total_outliers = outliers.sum().sum()
    st.write("Total jumlah outliers:", total_outliers)

    outlier_indices = np.where(outliers.any(axis=1))[0]
    no_outliers = data.drop(outlier_indices).reset_index(drop=True)
    st.write("Bentuk data awal:", data.shape)
    st.write("Bentuk data setelah menghapus outliers:", no_outliers.shape)

    scaler = MinMaxScaler()
    features_to_scale = ['BloodPressure', 'SkinThickness', 'BMI', 'Pregnancies', 'Glucose', 'Insulin', 'DiabetesPedigreeFunction', 'Age']
    no_outliers[features_to_scale] = scaler.fit_transform(no_outliers[features_to_scale])

    st.session_state["preprocessed_data"] = no_outliers
    st.write("Data setelah Normalisasi:", no_outliers.head(10))

# Split data page function
def Split_Data():
    st.title("Pembagian Data")
    data = st.session_state.get("preprocessed_data", load_data())

    Y = data['Outcome']
    X = data.drop(columns=['Outcome'])

    splits = {
        '90:10': train_test_split(X, Y, test_size=0.1, shuffle=True, random_state=42),
        '80:20': train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=42),
        '70:30': train_test_split(X, Y, test_size=0.3, shuffle=True, random_state=42)
    }

    st.session_state["splits"] = splits

    for split_name, (X_train, X_test, y_train, y_test) in splits.items():
        st.write(f"{split_name} Split:")
        st.write(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# Evaluation page function
def Evaluate_Models():
    st.title("Evaluasi Model")
    splits = st.session_state.get("splits", None)
    if splits is None:
        st.error("Silakan lakukan preprocessing dan pembagian data terlebih dahulu.")
        return

    models = {
        'Perceptron': Perceptron(max_iter=500),
        'GaussianNB': GaussianNB(),
        'KNeighbors': KNeighborsClassifier(n_neighbors=5)
    }

    learning_rates = np.linspace(0.01, 0.1, 10)
    param_grid = {'eta0': learning_rates}

    for split_name, (X_train, X_test, y_train, y_test) in splits.items():
        st.subheader(f"Evaluasi untuk {split_name} Split")
        for model_name, model in models.items():
            if model_name == 'Perceptron':
                grid_search = GridSearchCV(Perceptron(max_iter=500, random_state=42), param_grid, scoring='accuracy', cv=5)
                grid_search.fit(X_train, y_train)
                st.write(f"Perceptron Learning Rates and Accuracy:")
                results = pd.DataFrame(grid_search.cv_results_)
                st.write(results[['param_eta0', 'mean_test_score']])

                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                st.write(f"Best Learning Rate: {best_params['eta0']}")

                y_predict = best_model.predict(X_test)
            else:
                model.fit(X_train, y_train)
                y_predict = model.predict(X_test)

            akurasi = accuracy_score(y_test, y_predict)
            precision = precision_score(y_test, y_predict)
            recall = recall_score(y_test, y_predict)
            f1 = f1_score(y_test, y_predict)

            st.write(f"Model: {model_name}")
            st.write(f"Accuracy: {akurasi:.4f}")
            st.write(f"Precision: {precision:.4f}")
            st.write(f"Recall: {recall:.4f}")
            st.write(f"F1 Score: {f1:.4f}")

            # Confusion Matrix
            conf_matrix = confusion_matrix(y_test, y_predict)
            st.write(f"Confusion Matrix for {model_name}:")
            fig, ax = plt.subplots()
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f"Confusion Matrix: {model_name}")
            ax.set_xlabel("Predicted Labels")
            ax.set_ylabel("True Labels")
            st.pyplot(fig)

# Main function
def main():
    with st.sidebar:
        page = option_menu("Menu", ["Home", "Preprocessing", "Split Data", "Evaluasi Model"], default_index=0)

    if page == "Home":
        Home()
    elif page == "Preprocessing":
        Preprocessing()
    elif page == "Split Data":
        Split_Data()
    elif page == "Evaluasi Model":
        Evaluate_Models()

if __name__ == "__main__":
    st.set_page_config(page_title="Model Evaluasi Diabetes")
    main()
