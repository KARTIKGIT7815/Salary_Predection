from flask import Flask, render_template, request
import pickle


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('Task1.html')

@app.route('/predict', methods=['POST'])
def f2():
    name = request.form.get('name', '').strip()
    YearsExperience = int(request.form['YearsExperience'])
    print(request.form)
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    model = pickle.load(open('model_RF.pkl','rb'))
    
    q = [[YearsExperience]]
    q_scaled = scaler.transform(q)
    
    yp = model.predict(q_scaled)
    predicted_salary = str(round(yp[0],2))

    return render_template('Task2.html', name=name, salary=predicted_salary, experience=YearsExperience)

if __name__ == '__main__':
     #app.run(debug=True)
     app.run(host='0.0.0.0', port=8000)

