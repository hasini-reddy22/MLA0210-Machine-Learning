Python 3.14.3 (tags/v3.14.3:323c59a, Feb  3 2026, 16:04:56) [MSC v.1944 64 bit (AMD64)] on win32
Enter "help" below or click "Help" above for more information.
>>> 
= RESTART: C:/Users/hasin/AppData/Local/Programs/Python/Python314/Model Lab 1.py
Decision Tree:
{'MarketTrend': {'Up': 'Up', 'Down': 'Down'}}

Prediction for new record: Up
>>> 
= RESTART: C:/Users/hasin/AppData/Local/Programs/Python/Python314/Model lab 2.py
Prediction for new trading instance: Up
>>> 
= RESTART: C:/Users/hasin/AppData/Local/Programs/Python/Python314/Model lab 3.py

Calculating for class: Up
Prior P(Up) = 0.5000
P(MarketTrend=Up | Up) = 1.0000
P(Volume=Low | Up) = 0.4000
P(News=Negative | Up) = 0.4000
Posterior P(Up|X) = 0.080000

Calculating for class: Down
Prior P(Down) = 0.5000
P(MarketTrend=Up | Down) = 0.2500
P(Volume=Low | Down) = 0.4000
P(News=Negative | Down) = 0.6000
Posterior P(Down|X) = 0.030000

Final Prediction: Up
>>> 
= RESTART: C:/Users/hasin/AppData/Local/Programs/Python/Python314/Model lab 4.py
Trained Weights: [-5.76888709  0.23716022  1.12291563]
Trained Bias: 2.0574501520542645

Probability of Stock Moving Up: 0.9084
Predicted Movement: Up
