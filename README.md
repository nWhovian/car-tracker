## Vehicle detection and tracking system using the OpenVino toolkit

### Installation guide

Clone repository:
```
git clone https://github.com/nWhovian/car-tracker.git
cd path/to/car-tracker
```
Download file from [here](https://drive.google.com/uc?export=download&id=1ciX7cHqCh8lLFYI0HKkhC3r_fMirrlKk) and save it as
```/files/l_openvino_toolkit_p_2019.1.133.tgz```

Download the docker image:
```
docker-compose pull
```
Start and run the app:
```
docker-compose up
```
Access the running container:
```
 docker exec -it car_tracker_app_1 bash
```
Run the tracker:
```
python3.6 main.py 
```
Optional arguments: ```--video video.mkv --output out.mp4 --xml_path models/FP32/vehicle-detection-adas-0002.xml --bin_path models/FP32/vehicle-detection-adas-0002.bin```

The output file is located here by default: path/to/car-tracker/out.mp4
