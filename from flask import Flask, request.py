from flask import Flask, request
import video_file

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            try:
                with open("C:\\Users\\vaidu\\Downloads\\archive",'rb') as f:
                    pass
            except Exception as e:
                print(str(e))
                return 'file open failed'
            else:
                file.save(f)
                return 'file uploaded successfully'
    return 'file upload failed'

if __name__ == '__main__':
    app.run(debug=True)
    
