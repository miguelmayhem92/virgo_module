# Virgo Package

this package contains the utils and helper functions that used in virgo project

### how to use

istall using 

```
pip install virgo-modules
```

geting data:

```
obj = stock_eda_panel(stock_code = 'PEP', n_days = 20)
obj.get_data()
print(obj.df.shape)
```