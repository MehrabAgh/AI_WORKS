from flask import Flask,request,url_for,render_template
from markupsafe import escape

# create object flask for start server
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './upload_files'

# add route index for handling main page 
@app.route("/")
def homePage():            
    return f'''
    <b>hi web</b>
    <link rel='stylesheet' href='{url_for('static', filename='style.css')}'>
    <b>hi web1</b>
    <b>hi web2</b>
    '''

# add subroute
@app.route("/salam/")
def testPage():
    return f"<p>hello</p>"

# create pass variable in url and use 'url_for' for dynamically urls
# use escape for get values with security
@app.route("/<int:name>")
def aboutPage(name):
    return f"<a href={url_for('testPage')}>{escape(name)}</h>"

# implement render template html file
@app.route("/contact/<int:name>" , methods =["get","post"])
def contactPage(name):    
    if(request.method == "POST"):
        file = request.files['file']
        file.save(f'/upload_files/{file.name}')
        print(file)
    return render_template('index.jinja',aa=name)

# http request code and pass data
@app.post('/login')
def SendData():
    print('bye login t1')
    print(request.form['myApp'])
    return request.form['myApp']
@app.get('/login')
def getdata():
    print('hi login t1', request.args.get('web',None))
    return ''
@app.route('/login',methods = ['get','post'])
def MainRequest():
    if(request.method == "get"):
        return gd()
    else: 
        return sd()    
def gd():
    print('hi login t2')
    return ''
def sd():
    print('bye login t2')
    return ''

# run application
app.run()