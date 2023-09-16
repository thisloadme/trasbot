# Trasbot - Traspac Chatbot
a rule based chatbot that uses LSTM to detect labels from user input and responds accordingly.

# Web Version Installation Guide
1. Install the required packages by running
```
pip install -r requirements.txt
```

2. Adjust the port for the chatbot server in `app.py`

3. Run the chatbot server by running
```
python app.py
```

if you want to run in the background on linux, run
```
nohup python app.py >/dev/null 2>&1 &
```
> Replace python with python3 according to the installed python

4. Adjust the server host and port in the `app.js` file in the `standalone` folder

5. Run the chatbot UI by running `index.html` in the `standalone` folder 5.

# Telegram Version Installation Guide
1. Rename `.env.example` to `.env`

2. Fill `TELEGRAM_API_KEY` with API_KEY of Telegram chatbot

3. Run the telegram bot service by running
```
python telegram_bot.py
```

if you want it to run in the background on Linux, run
```
nohup python telegram_bot.py >/dev/null 2>&1 &
```
> Replace python with python3 according to the installed python