#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle


# In[17]:


app = Flask(__name__)
model = pickle.load(open(r'C:\Users\prash\Desktop\data\diabetes prediction\diabetes.pkl','rb'))


# In[18]:


@app.route('/')
def home():
    return render_template('index.html')


# In[19]:


@app.route('/predict',methods=['POST'])
def predict():
    
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    output = round(prediction[0], 2)

        
    return render_template('index.html',prediction_text = 'You have {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)


# In[ ]:





# In[ ]:




