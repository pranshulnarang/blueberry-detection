from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="openai_detection")
trainer.setTrainConfig(object_names_array=["mature","intermediate","young"], batch_size=32, num_experiments=300)
trainer.trainModel()
