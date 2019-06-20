from flask import Flask
import routes

ml = Flask('ml_api')
ml.register_blueprint(routes.model_predictor)
ml.run(host='0.0.0.0', port=5005)