# iris_prediction_server.py
import grpc
from concurrent import futures
import time
import joblib
from sklearn import datasets
from .generated import iris_demo_pb2, iris_demo_pb2_grpc


_ONE_DAY_IN_SECONDS = 60 * 60 * 24


# predict_iris.py


class PredictServicer(iris_demo_pb2_grpc.PredictServicer):
    def predict_iris_target(self, request, context):
        trained_model = joblib.load('/home/mlpy/PyProj/pys/IrisClassifier.pkl')
        response = iris_demo_pb2.IrisPredictResponse()
        response.target = trained_model.predict(
            request.sepal_length,
            request.sepal_width,
            request.petal_length,
            request.petal_width,
        )
        return response  ##not sure


def run():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    iris_demo_pb2_grpc.add_PredictServicer_to_server(PredictServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)
