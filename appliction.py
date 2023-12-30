from flask import Flask,request,render_template,jsonify
from src.pipelines.prediction_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application

@app.route('/')
def home_page():
    return render_template('form.html')

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data=CustomData(
            
            qty_slash_url=float(request.form.get('qty_slash_url')),
            length_url= float(request.form.get('length_url')),
            qty_dot_domain=float(request.form.get('qty_dot_domain')),
            qty_dot_directory= float(request.form.get('qty_dot_directory')),
            qty_hyphen_directory= float(request.form.get('qty_hyphen_directory')),
            qty_underline_directory = float(request.form.get('qty_underline_directory')),
            
            asn_ip=float(request.form.get('asn_ip')),
            time_domain_activation=float(request.form.get('time_domain_activation')),
            time_domain_expiration=float(request.form.get('time_domain_expiration')),
            ttl_hostname=float(request.form.get('ttl_hostname'))
            
        )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        results=round(pred[0],2)

        return render_template('form.html',final_result=results)
    

if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)