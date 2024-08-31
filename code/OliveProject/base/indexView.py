import ml_model.App_OliveClass as ao

from flask import Flask , request , render_template
from werkzeug.utils import secure_filename
import os

app = Flask(__name__ , template_folder='./template')
app.config['STATIC_FOLDER'] = 'static'
app.config['UPLOAD'] = '/AI_WORKS/code/OliveProject/base/static/upload'

dataFile = list()

@app.route('/about')
def about():
    return render_template('about.jinja')

@app.route('/' , methods = ['GET' , 'POST'])
def index():    
    if(request.method == 'POST'):
        f = request.files['image']
        filename = secure_filename(str(f.filename))
        dirImg = os.path.join(app.config['UPLOAD'],filename)        
        
        if(len(dataFile) > 0):
            for i in range(len(dataFile)):
                os.remove(dataFile[i])
                dataFile.pop(i)
        else: dataFile.append(dirImg)
        
        f.save(dirImg)        
        t = ao.ProcessImage(dirImg)
                       
        return render_template('index.jinja' , err = t['err'] , acc =t['acc']
            , classType=t['className'], enable="1" , imageName=f.filename)
    return render_template('index.jinja' , err = '' , acc ='' , classType='')

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404
app.run()
