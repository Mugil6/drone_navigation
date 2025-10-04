from roboflow import Roboflow
from ultralytics import YOLO

rf = Roboflow(api_key="lUhtTemuhbe8FBNxdw6T")
project = rf.workspace("uav123").project("uav123")
dataset = project.version(1).download("yolov8")


model = YOLO("yolov8n.pt") 
model.train(data=dataset.location + "/data.yaml", epochs=50, imgsz=640)


metrics = model.val()
print(metrics)

model.export(format="pt")
