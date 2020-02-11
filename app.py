from flask import Flask
from flask_restful import Resource, Api
from src.palmcanny import PalmCanny
from src.morph import MorphCanny
from src.sobel import Sobel
from src.plate import Plate

app = Flask(__name__)
api = Api(app)

api.add_resource(PalmCanny, '/canny')
api.add_resource(MorphCanny, '/morphological')
api.add_resource(Sobel, '/sobel')
api.add_resource(Plate, '/plate')

if __name__ == '__main__':
    app.run(host='0.0.0.0')