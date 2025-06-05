import warnings
import logging
import os
from sklearn.exceptions import ConvergenceWarning

# Suppress all warnings
warnings.filterwarnings('ignore')

# Suppress specific library warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# Set logging level to ERROR only
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger('werkzeug').setLevel(logging.ERROR)

# Suppress matplotlib warnings
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
matplotlib.rcParams['figure.max_open_warning'] = 0

# Suppress pandas warnings
import pandas as pd
pd.options.mode.chained_assignment = None

from flask import Flask

def create_app():
    app = Flask(__name__)
    
    # Register blueprints
    from app.routes.main import bp as main_bp
    app.register_blueprint(main_bp)
    
    return app 