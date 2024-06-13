import os
import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo
import matplotlib.pyplot as plt
import pyreadstat
import json
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, render_template, send_file, redirect, url_for

# Configurar R_HOME directamente en el script
os.environ['R_HOME'] = r'C:\Program Files\R\R-4.4.0'  # Asegúrate de ajustar la versión de R

# Activar la conversión automática entre pandas DataFrame y R data.frame
pandas2ri.activate()

# Importar paquetes R necesarios
lavaan = importr('lavaan')
base = importr('base')

app = Flask(__name__)

# Función para cargar datos desde diferentes formatos de archivo
def load_data(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == '.csv':
        df = pd.read_csv(file_path)
    elif file_extension == '.xlsx' or file_extension == '.xls':
        df = pd.read_excel(file_path)
    elif file_extension == '.txt':
        df = pd.read_csv(file_path, delimiter='\t')
    elif file_extension == '.sav':
        df, meta = pyreadstat.read_sav(file_path)
    elif file_extension == '.json':
        df = pd.read_json(file_path)
    elif file_extension == '.dta':
        df = pd.read_stata(file_path)
    elif file_extension == '.sas7bdat':
        df = pd.read_sas(file_path)
    elif file_extension == '.h5':
        df = pd.read_hdf(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")
    return df

# Función para manejar datos faltantes
def handle_missing_data(df):
    df.fillna(df.mean(), inplace=True)
    return df

# Función para calcular la prueba de esfericidad de Bartlett
def bartlett_test(df):
    chi_square_value, p_value = calculate_bartlett_sphericity(df)
    return chi_square_value, p_value

# Función para calcular el índice de adecuación muestral de KMO
def kmo_test(df):
    kmo_all, kmo_model = calculate_kmo(df)
    return kmo_all, kmo_model

# Función para realizar el análisis paralelo manual
def parallel_analysis(df, n_resamples=1000, random_state=0):
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
def perform_efa(df, n_factors, rotation='varimax'):
    fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation)
    fa.fit(df)
    factor_loadings = pd.DataFrame(fa.loadings_, index=df.columns)
    return factor_loadings, fa.get_eigenvalues()[0]

# Función para automatizar la generación del modelo de CFA
def generate_cfa_model(factor_loadings, threshold=0.4):
    factors = {}
    for item, loadings in factor_loadings.iterrows():
        for factor, loading in loadings.items():
            if abs(loading) > threshold:
                if factor not in factors:
                    factors[factor] = []
                factors[factor].append(item)
    model = []
    for factor, items in factors.items():
        factor_name = f"F{factor+1}"
        items_str = " + ".join(items)
        model.append(f"{factor_name} =~ {items_str}")
    return "\n".join(model)

# Función para realizar CFA
def perform_cfa(df, model):
    r_df = pandas2ri.py2rpy(df)
    cfa_model = robjects.StrVector(model.split("\n"))
    print(f"Model passed to lavaan:\n{model}")
    cfa_result = lavaan.cfa(cfa_model, data=r_df)
    summary = base.summary(cfa_result, fit_measures=True)
    return summary

# Función para evaluar la validez del instrumento
def evaluate_validity(bartlett_p_value, kmo_model, factor_loadings):
    if (bartlett_p_value > 0.05):
        return "La prueba de esfericidad de Bartlett no es significativa. Los datos podrían no ser adecuados para el análisis factorial."
    if (kmo_model < 0.6):
        return "El índice de adecuación muestral de KMO es bajo. Los datos podrían no ser adecuados para el análisis factorial."

    valid_factors = (factor_loadings.abs() > 0.4).sum(axis=0)
    if (valid_factors >= 3).all():
        return "El instrumento es válido en base a las cargas factoriales."
    else:
        return "El instrumento no es válido en base a las cargas factoriales."

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)
            n_folds = int(request.form.get('n_folds', 5))

            df = load_data(file_path)
            df = handle_missing_data(df)

            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(bartlett_test, df): "Bartlett Test",
                    executor.submit(kmo_test, df): "KMO Test"
                }
                results = {}
                for future, test_name in futures.items():
                    try:
                        results[test_name] = future.result()
                    except Exception as exc:
                        return f"{test_name} generated an exception: {exc}"

            chi_square_value, p_value = results.get("Bartlett Test", (None, None))
            kmo_all, kmo_model = results.get("KMO Test", (None, None))

            _, eigenvalues_real = perform_efa(df, df.shape[1])
            eigenvalues_random = parallel_analysis(df)

            plt.plot(range(1, df.shape[1] + 1), eigenvalues_real, marker='o', label='Eigenvalores Reales')
            plt.plot(range(1, df.shape[1] + 1), eigenvalues_random, marker='o', label='Eigenvalores Aleatorios')
            plt.xlabel('Número de Factores')
            plt.ylabel('Eigenvalor')
            plt.title('Análisis Paralelo')
            plt.legend()
            plt.savefig('static/plot.png')
            plt.close()

            n_factors = sum(eigenvalues_real > eigenvalues_random)
            n_factors_cv = cross_validation(df, n_factors, n_folds)

            factor_loadings, _ = perform_efa(df, n_factors_cv)
            cfa_model = generate_cfa_model(factor_loadings)
            summary = perform_cfa(df, cfa_model)

            validity = evaluate_validity(p_value, kmo_model, factor_loadings)

            results_text = (f"Bartlett's test: chi_square_value = {chi_square_value}, p_value = {p_value}\n"
                            f"KMO: {kmo_model}\n"
                            f"El número óptimo de factores es: {n_factors}\n"
                            f"El número más común de factores en validación cruzada es: {n_factors_cv}\n"
                            f"{factor_loadings}\n"
                            f"Generated CFA Model:\n{cfa_model}\n"
                            f"{summary}\n"
                            f"{validity}")

            return render_template('index.html', results=results_text, plot_url=url_for('static', filename='plot.png'))

    return render_template('index.html')

def cross_validation(df, n_factors, k=5):
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=k, shuffle=True, random_state=0)
    results = []

    for train_index, test_index in kf.split(df):
        train_df, test_df = df.iloc[train_index], df.iloc[test_index]
        _, eigenvalues_real = perform_efa(train_df, n_factors)
        eigenvalues_random = parallel_analysis(train_df)

        n_factors_cv = sum(eigenvalues_real > eigenvalues_random)
        results.append(n_factors_cv)

    most_common_factors = max(set(results), key=results.count)
    print(f"El número más común de factores en validación cruzada es: {most_common_factors}")
    return most_common_factors

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
