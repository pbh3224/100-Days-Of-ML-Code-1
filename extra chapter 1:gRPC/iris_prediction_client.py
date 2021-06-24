# iris_prediction_client.py
import grpc
from .generated import iris_demo_pb2, iris_demo_pb2_grpc


channel = grpc.insecure_channel('localhost:50051')
stub = iris_demo_pb2_grpc.PredictStub(channel)
requestPrediction = iris_demo_pb2.IrisPredictRequest(
    sepal_length=7.233, sepal_width=4.652, petal_length=7.39, petal_width=0.324
)
responsePrediction = stub.predict_iris_target(requestPrediction)
print('The prediction is :', responsePrediction.target)