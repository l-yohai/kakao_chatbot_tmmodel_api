import tensorflow.keras
import flask
import urllib.request
from PIL import Image, ImageOps
import numpy as np

app = flask.Flask(__name__)
model = None


def load_model():

    global model
    model = tensorflow.keras.models.load_model('keras_model.h5')


@app.route("/api/predict", methods=["POST"])
def api_predict():

    # UserRequest 중 발화를 req에 parsing.
    req = flask.request.get_json()
    req = req['userRequest']['utterance']

	# 이미지 전처리 - 발화가 jpg, png 확장자일 때만 실행
    if 'jpg' in req or 'png' in req:
        np.set_printoptions(suppress=True)
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        urllib.request.urlretrieve(req, 'img')
        image = Image.open('img').convert('RGB')
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array
        prediction = model.predict(data)

        output = np.argmax(prediction, axis=-1)

        if (output == 0):
            msg = "개입니다!"
        else:
            msg = "고양이입니다!"

		# basic card format
		res = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "basicCard": {
                            "title": msg,
                            "description": "",
                            "thumbnail": {
                                "imageUrl": req
                            },
                            "buttons": [
                                {
                                    "action":  "webLink",
                                    "label": "사진보기",
                                    "webLinkUrl": req
                                }
                            ]
                        }
                    }
                ]
            }
        }
        
    else:
    	# simple text format
        res = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": "사진을 보내주세요"
                        }
                    }
                ]
            }
        }
    print(res)
    return flask.jsonify(res)


if __name__ == "__main__":
    print("* Loading Keras model and Flask starting server...")
    print("please wait until server has fully started")
    load_model()
    app.run(host='0.0.0.0')
