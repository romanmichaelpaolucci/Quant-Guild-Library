import openai


import alpaca_trade_api as tradeapi

key = "PKGDU2MIBWB8TAE9CDQ0"
secret_key = "el7A7EF5d5y3ILvYr17O30fwZkPKmUZfcZEVxUov"
BASE_URL = "https://paper-api.alpaca.markets/"

api = tradeapi.REST(key, secret_key, BASE_URL, api_version="v2")




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


analysis

