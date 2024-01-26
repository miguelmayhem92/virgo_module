from virgo_modules.src.ticketer_source import stock_eda_panel
    
obj = stock_eda_panel(stock_code = 'PEP', n_days = 20)
obj.get_data()
print(obj.df.shape)