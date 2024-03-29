import virgo_modules as vmod 

from virgo_modules.src.re_utils import calculate_cointegration
from virgo_modules.src.ticketer_source import stock_eda_panel
from virgo_modules.src.edge_utils import produce_model_wrapper


obj = stock_eda_panel(stock_code = 'PEP', n_days = 15000,data_window = '15y' )
obj.get_data()
print(obj.df.shape)