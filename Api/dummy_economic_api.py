from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/api/economic-indicators', methods=['GET'])
def economic_indicators():
    return jsonify({
        "source": "Dummy Economic Indicators API",
        "data": [
            {"region": "North", "unemployment_rate": 5.2, "inflation_rate": 4.8, "interest_rate": 6.5},
            {"region": "South", "unemployment_rate": 4.7, "inflation_rate": 4.5, "interest_rate": 6.5},
            {"region": "East", "unemployment_rate": 5.8, "inflation_rate": 5.1, "interest_rate": 6.5},
            {"region": "West", "unemployment_rate": 4.9, "inflation_rate": 4.6, "interest_rate": 6.5}
        ]
    })

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
