1. What is this ?
機械学習用モデル、及びエクセルデータを呼び込むためのモジュール

2. Directory Tree

Machine-Learning
├── newral_net.py
├── excel_reader.py
├── weights
│     ├── W1.pickle
│     ├── b1.pickle
│     ├── W2.pickle
│     └── b2.pickle
└── excels
       └── data.xlsx

3. How to use this?
newral_net.pyを立ち上げれば、あとは自動で学習してくれます。
学習後の重みの保存は「NewralNet」クラスの「save_weights」メソッドを呼び出してください。