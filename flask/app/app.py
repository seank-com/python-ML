import time
import tensorflow as tf
import cv2
import numpy as np
from label_image import load_graph, load_labels
from flask import Response
from flask import request
from flask import Flask

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index_get():
    resp = Response(response='{"test": "Foo"}',
        status=200,
        mimetype="application/json")
    return resp


def predict(img):
    graph_file = 'output_graph.pb'
    label_file = 'output_labels.txt'

    graph = load_graph(graph_file)
    input_name = "import/input"
    output_name = "import/final_result"
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)
    labels = load_labels(label_file)

    with tf.Session(graph=graph) as sess:
        image = np.asarray(bytearray(img), dtype="uint8")
        frame = cv2.imdecode(image, cv2.IMREAD_COLOR)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # img = get_image()

        t,r = preprocess(frame, 224, 224)
        t1 = time.time()

        results = sess.run(output_operation.outputs[0],
                        {input_operation.outputs[0]: t})
        results = np.squeeze(results)

        top_k = results.argsort()[-5:][::-1]
        Top1 = top_k[0]
        return labels[Top1], results[Top1]


@app.route('/', methods=['POST'])
def process():
    resp = Response(response='{"tes":"poo"}',status=201,  mimetype="application/json")
    if request.headers['Content-Type'] == 'application/octet-stream':
        data = request.data
        l = len(data)

        prediction, percent = predict(data)
        resp = Response(response='{"precidtion":"'+prediction+'"}',
                    status=200,
                    mimetype="application/json")
    return resp
