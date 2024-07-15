import ml_model.App_OliveClass as ao

from flask import Flask , request , render_template
from werkzeug.utils import secure_filename
import os

app = Flask(__name__ , template_folder='./template')
app.config['STATIC_FOLDER'] = 'static'
app.config['UPLOAD'] = '/AI_WORKS/code/OliveProject/base/static'

dataFile = list()

@app.route('/' , methods = ['GET' , 'POST'])
def index():    
    if(request.method == 'POST'):
        f = request.files['image']
        filename = secure_filename(str(f.filename))
        dirImg = os.path.join(app.config['UPLOAD'],filename)        
        
        if(len(dataFile) > 0):
            os.remove(dataFile[0])
            dataFile.pop(0)
        dataFile.append(dirImg)
        
        f.save(dirImg)        
        t = ao.ProcessImage(dirImg)
                       
        return render_template('index.jinja' , err = t['err'] , acc =t['acc']
            , classType=t['className'], enable="1" , imageName=f.filename)
    return render_template('index.jinja' , err = '' , acc ='' , classType='')


app.run()
