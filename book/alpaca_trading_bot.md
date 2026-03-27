```python
import alpaca_trade_api as tradeapi

key = "KEY_ALPCA"
secret_key = "SECRET_KEY_ALPCA"
BASE_URL = "https://paper-api.alpaca.markets/"

api = tradeapi.REST(key, secret_key, BASE_URL, api_version="v2")
```


```python
def get_data(symbol):
    try:
        barset = api.get_latest_trade(symbol)
        return {"price":barset.price}
    except Exception as e:
        return {"price":-1}

get_data("AAPL")
```




    {'price': 213.88}




```python
def get_max_entry_price(symbol):
    try:
        orders = api.list_orders(status="filled", limit=50)
        prices = [float(order.filled_avg_price) for order in orders if order.filled_avg_price]
        return max(prices) if prices else -1
    except Exception as e:
        return 0
get_max_entry_price("AAPL")
```




    213.99


