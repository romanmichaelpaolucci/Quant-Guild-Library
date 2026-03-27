```python
import openai
```


```python
import alpaca_trade_api as tradeapi

key = "PKGDU2MIBWB8TAE9CDQ0"
secret_key = "el7A7EF5d5y3ILvYr17O30fwZkPKmUZfcZEVxUov"
BASE_URL = "https://paper-api.alpaca.markets/"

api = tradeapi.REST(key, secret_key, BASE_URL, api_version="v2")
```


```python


def analyze_message(message):
    portfolio_data = fetch_portfolio()
    open_orders = fetch_open_orders()

    pre_prompt = f"""
    You are an AI Portfolio Manage responsible for analyzing my portfolio.
    Your tasks are the following:
    1.) Evaluate risk exposures of my current holdings
    2.) Analyze my open limit orders and their potential impact
    3.) provide insights into portfolio health, diversification, trade adj. etc.
    4.) Speculate on the market outlook based on current market conditions
    5.) Identify potential market risks and suggest risk management strategies

    Here is my portfolio: {portfolio_data}

    Here are my open orders {open_orders}

    Overall, answer the following question with priority having that background: {message}
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role":"system", "content":pre_prompt}],
        api_key = "SECRET_KEY_OPENAI"
    )
    return response['choices'][0]['message']['content']

analysis = analyze_message("How is my portfolio doing?")
```


```python
analysis
```




    "As an AI Portfolio Manager, I have analyzed your portfolio as per the details provided.\n\n1) Risk Exposures: Currently, your portfolio consists of a single stock, 'AAPL'. This implies that your portfolio lacks diversification and hence is exposed to high unsystematic risk. Unsystematic risk is the risk associated with a specific stock and can be mitigated by diversification. Having a concentrated position in one stock can lead to significant losses if the price of the AAPL stock decreases.\n\n2) Open Limit Orders: You currently do not have any open orders.\n\n3) Portfolio Health & Diversification: As mentioned earlier, your portfolio is not diversified as it contains only one stock. A diversified portfolio typically includes a mix of equities, bonds, commodities, and other assets. This helps to decrease the unsystematic risk and smooth out returns.\n\n4) Market Outlook: As an AI, I do not speculate on market conditions. My analysis is based on historical data, current market conditions, and formulated algorithms.\n\n5) Market Risks and Risk Management Strategies: A significant risk in a single stock portfolio is that the portfolio's value will be significantly impacted by the price movement of that single stock. One risk management strategy, in this case, would be to diversify your portfolio. This may include buying stocks of various companies in different industries or investing in mutual funds. \n\nIn response to your question about the performance of your portfolio: Based on the information provided, your portfolio seems to be performing well currently. The AAPL stock's current price has slightly increased from your entry price leading to a marginal unrealized profit. However, due to a lack of diversification, the health of your portfolio is at high risk as it may be significantly affected by the price movements of AAPL stock. It is recommended to diversify your investment in order to manage and mitigate risks."


