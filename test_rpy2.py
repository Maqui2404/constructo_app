import os

# Configurar R_HOME directamente en el script
os.environ['R_HOME'] = r'C:\Program Files\R\R-4.4.0'  # Ajusta la versión de R según sea necesario

import rpy2.robjects as robjects

# Verificar la versión de R
version = robjects.r('version')
print(version)
