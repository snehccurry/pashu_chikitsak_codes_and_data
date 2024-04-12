from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)



#TODO: improve the already existing mode if possible with new test images for more accuracy.



#model.train(data='D:\\TEST\\final_bca_project\\pythonProject\\test\\animal_dataset', epochs=50, imgsz=64)

model.train(data='D:\\TEST\\final_bca_project\\pythonProject\\test\\animal_dataset', epochs=16, imgsz=64)



