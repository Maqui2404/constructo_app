import os

from factorial import cross_validation

# Configurar R_HOME directamente en el script
r_home = r'C:\Program Files\R\R-4.4.0'
if not os.environ.get('R_HOME'):
    os.environ['R_HOME'] = r_home

from shiny import App, render, ui
import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo
import matplotlib.pyplot as plt
import pyreadstat
import json
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from concurrent.futures import ThreadPoolExecutor, as_completed

# Activar la conversión automática entre pandas DataFrame y R data.frame
pandas2ri.activate()

# Importar paquetes R necesarios
lavaan = importr('lavaan')
base = importr('base')

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

# Interfaz de usuario
app_ui = ui.page_fluid(
    ui.panel_title("Validación de Instrumentos"),
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.input_file("file_input", "Cargar Archivo de Datos"),
            ui.input_numeric("n_folds", "Número de Folds para Validación Cruzada", 5),
            ui.input_action_button("run_analysis", "Ejecutar Análisis")
        ),
        ui.panel_main(
            ui.output_text_verbatim("output"),
            ui.output_plot("plot")
        )
    )
)

# Lógica del servidor
def server(input, output, session):
    @output
    @render.text
    def output_text():
        if input.run_analysis():
            file_info = input.file_input()
            if not file_info:
                return "Por favor, cargue un archivo de datos."

            file_path = file_info[0]["datapath"]
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
            plt.savefig('plot.png')
            plt.close()

            n_factors = sum(eigenvalues_real > eigenvalues_random)
            n_factors_cv = cross_validation(df, n_factors, input.n_folds())

            factor_loadings, _ = perform_efa(df, n_factors_cv)
            cfa_model = generate_cfa_model(factor_loadings)
            summary = perform_cfa(df, cfa_model)

            validity = evaluate_validity(p_value, kmo_model, factor_loadings)

            return (f"Bartlett's test: chi_square_value = {chi_square_value}, p_value = {p_value}\n"
                    f"KMO: {kmo_model}\n"
                    f"El número óptimo de factores es: {n_factors}\n"
                    f"El número más común de factores en validación cruzada es: {n_factors_cv}\n"
                    f"{factor_loadings}\n"
                    f"Generated CFA Model:\n{cfa_model}\n"
                    f"{summary}\n"
                    f"{validity}")

    @output
    @render.plot
    def plot_output():
        if input.run_analysis():
            img = plt.imread('plot.png')
            plt.imshow(img)
            plt.axis('off')
            return plt.gcf()

# Crear la aplicación
app = App(app_ui, server)

# Ejecutar la aplicación
app.run()
