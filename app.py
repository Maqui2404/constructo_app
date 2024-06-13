import os
import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import KFold

import io

# Función para cargar datos desde diferentes formatos de archivo
@st.cache_data
def load_data(file_upload):
    """
    Carga datos desde diferentes formatos de archivo.

    Args:
        file_upload (UploadedFile): Archivo cargado por el usuario.

    Returns:
        pandas.DataFrame: DataFrame con los datos cargados.
    """
    try:
        if file_upload is not None:
            file_extension = os.path.splitext(file_upload.name)[1].lower()
            if file_extension == '.csv':
                df = pd.read_csv(io.BytesIO(file_upload.getvalue()))
            elif file_extension == '.xlsx' or file_extension == '.xls':
                df = pd.read_excel(io.BytesIO(file_upload.getvalue()))
            elif file_extension == '.txt':
                df = pd.read_csv(io.BytesIO(file_upload.getvalue()), delimiter='\t')
            elif file_extension == '.sav':
                df = pd.read_spss(io.BytesIO(file_upload.getvalue()))
            elif file_extension == '.json':
                df = pd.read_json(io.BytesIO(file_upload.getvalue()))
            elif file_extension == '.dta':
                df = pd.read_stata(io.BytesIO(file_upload.getvalue()))
            elif file_extension == '.sas7bdat':
                df = pd.read_sas(io.BytesIO(file_upload.getvalue()))
            elif file_extension == '.h5':
                df = pd.read_hdf(io.BytesIO(file_upload.getvalue()))
            else:
                raise ValueError(f"Formato de archivo no soportado: {file_extension}")
            return df
        else:
            return None
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")
        return None

# Función para manejar datos faltantes
def handle_missing_data(df):
    """
    Maneja los datos faltantes en el DataFrame reemplazándolos por la media.

    Args:
        df (pandas.DataFrame): DataFrame con datos.

    Returns:
        pandas.DataFrame: DataFrame con datos faltantes reemplazados por la media.
    """
    df.fillna(df.mean(), inplace=True)
    return df

# Función para calcular la prueba de esfericidad de Bartlett
def bartlett_test(df):
    """
    Calcula la prueba de esfericidad de Bartlett.

    Args:
        df (pandas.DataFrame): DataFrame con datos.

    Returns:
        tuple: Valor Chi-cuadrado y valor p de la prueba de Bartlett.
    """
    chi_square_value, p_value = calculate_bartlett_sphericity(df)
    return chi_square_value, p_value

# Función para calcular el índice de adecuación muestral de KMO
def kmo_test(df):
    """
    Calcula el índice de adecuación muestral de KMO.

    Args:
        df (pandas.DataFrame): DataFrame con datos.

    Returns:
        tuple: Valor KMO global y KMO por cada variable.
    """
    kmo_all, kmo_model = calculate_kmo(df)
    return kmo_all, kmo_model

# Función para realizar el análisis paralelo manual
@st.cache_data
def parallel_analysis(df, n_resamples=1000, random_state=0):
    """
    Realiza el análisis paralelo manual para determinar el número óptimo de factores.

    Args:
        df (pandas.DataFrame): DataFrame con datos.
        n_resamples (int, optional): Número de remuestreos para el análisis paralelo. Predeterminado a 1000.
        random_state (int, optional): Semilla para el generador de números aleatorios. Predeterminado a 0.

    Returns:
        numpy.ndarray: Eigenvalores aleatorios obtenidos del análisis paralelo.
    """
    np.random.seed(random_state)
    random_eigenvalues = []

    for _ in range(n_resamples):
        random_data = np.random.normal(size=df.shape)
        fa = FactorAnalyzer(n_factors=df.shape[1], rotation=None)
        fa.fit(random_data)
        random_eigenvalues.append(fa.get_eigenvalues()[0])

    random_eigenvalues = np.mean(random_eigenvalues, axis=0)
    return random_eigenvalues

# Función para realizar el análisis factorial exploratorio (EFA)
@st.cache_data
def perform_efa(df, n_factors, rotation='varimax'):
    """
    Realiza el análisis factorial exploratorio (EFA).

    Args:
        df (pandas.DataFrame): DataFrame con datos.
        n_factors (int): Número de factores a extraer.
        rotation (str, optional): Método de rotación para los factores. Predeterminado a 'varimax'.

    Returns:
        tuple: DataFrame con las cargas factoriales y eigenvalores reales.
    """
    fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation)
    fa.fit(df)
    factor_loadings = pd.DataFrame(fa.loadings_, index=df.columns)
    return factor_loadings, fa.get_eigenvalues()[0]

# Función para evaluar la validez del instrumento
def evaluate_validity(bartlett_p_value, kmo_model, factor_loadings):
    """
    Evalúa la validez del instrumento basado en los resultados de la prueba de Bartlett,
    el índice KMO y las cargas factoriales.

    Args:
        bartlett_p_value (float): Valor p de la prueba de Bartlett.
        kmo_model (float): Valor KMO por variable.
        factor_loadings (pandas.DataFrame): DataFrame con las cargas factoriales.

    Returns:
        bool: True si el instrumento es válido, False en caso contrario.
    """
    if bartlett_p_value > 0.05:
        st.warning("La prueba de esfericidad de Bartlett no es significativa. Los datos podrían no ser adecuados para el análisis factorial.")
        return False
    if kmo_model < 0.6:
        st.warning("El índice de adecuación muestral de KMO es bajo. Los datos podrían no ser adecuados para el análisis factorial.")
        return False

    valid_factors = (factor_loadings.abs() > 0.4).sum(axis=0)
    if (valid_factors >= 3).all():
        st.success("El instrumento es válido en base a las cargas factoriales.")
        return True
    else:
        st.warning("El instrumento no es válido en base a las cargas factoriales.")
        return False

# Función para realizar validación cruzada
def cross_validation(df, n_factors, k=5):
    """
    Realiza la validación cruzada para determinar el número más estable de factores.

    Args:
        df (pandas.DataFrame): DataFrame con datos.
        n_factors (int): Número inicial de factores.
        k (int, optional): Número de iteraciones para la validación cruzada. Predeterminado a 5.

    Returns:
        int: Número más común de factores obtenido en la validación cruzada.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=0)
    results = []

    for (train_index, test_index) in kf.split(df):
        train_df, test_df = df.iloc[train_index], df.iloc[test_index]
        _, eigenvalues_real = perform_efa(train_df, n_factors)
        eigenvalues_random = parallel_analysis(train_df)

        n_factors_cv = sum(eigenvalues_real > eigenvalues_random)
        results.append(n_factors_cv)

    most_common_factors = max(set(results), key=results.count)
    st.info(f"El número más común de factores en validación cruzada es: {most_common_factors}")
    return most_common_factors

# Aplicación Streamlit
def main():
    st.title("Análisis Factorial Exploratorio y Confirmatorio")

    # Cargar archivo
    file_upload = st.file_uploader("Selecciona un archivo de datos", type=['csv', 'xlsx', 'xls', 'txt', 'sav', 'json', 'dta', 'sas7bdat', 'h5'])

    if file_upload is not None:
        st.subheader("Carga de datos")
        # Cargar datos
        df = load_data(file_upload)
        if df is not None:
            df = handle_missing_data(df)

            st.subheader("Pruebas preliminares")
            # Realizar pruebas preliminares
            chi_square_value, p_value = bartlett_test(df)
            kmo_all, kmo_model = kmo_test(df)

            st.write(f"Bartlett's test: chi_square_value = {chi_square_value:.2f}, p_value = {p_value:.4f}")
            st.write(f"KMO: {kmo_model:.2f}")

            st.subheader("Análisis Paralelo")
            # Realizar EFA
            _, eigenvalues_real = perform_efa(df, df.shape[1])
            eigenvalues_random = parallel_analysis(df)

            # Graficar los eigenvalores reales y aleatorios
            fig, ax = plt.subplots()
            ax.plot(range(1, df.shape[1] + 1), eigenvalues_real, marker='o', label='Eigenvalores Reales')
            ax.plot(range(1, df.shape[1] + 1), eigenvalues_random, marker='o', label='Eigenvalores Aleatorios')
            ax.set_xlabel('Número de Factores')
            ax.set_ylabel('Eigenvalor')
            ax.set_title('Análisis Paralelo')
            ax.legend()
            st.pyplot(fig)

            st.subheader("Determinación del número óptimo de factores")
            # Determinar el número óptimo de factores
            n_factors = sum(eigenvalues_real > eigenvalues_random)
            st.info(f"El número óptimo de factores es: {n_factors}")

            st.subheader("Validación Cruzada")
            # Realizar validación cruzada para determinar el número más estable de factores
            n_folds = st.slider("Número de iteraciones para validación cruzada", 2, 10, 5)
            n_factors_cv = cross_validation(df, n_factors, n_folds)

            st.subheader("Análisis Factorial Exploratorio (EFA)")
            # Realizar EFA con el número óptimo de factores
            rotation = st.selectbox("Método de rotación", ['varimax', 'quartimax', 'equamax', 'oblimin'])
            factor_loadings, _ = perform_efa(df, n_factors_cv, rotation)
            st.write("Cargas factoriales:")
            st.write(factor_loadings)

            st.subheader("Evaluación de validez")
            # Evaluar la validez
            is_valid = evaluate_validity(p_value, kmo_model, factor_loadings)

if __name__ == "__main__":
    main()
    


