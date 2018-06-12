# Aeroarms Crawler Detector

### Install
```
roscd && cd ../src/iri/
git clone https://github.com/albertpumarola/crawler_detector.git
cd crawler_detector
pip install -r requirements.txt
```
### Gate
The gate node to select pose source:
```
roscd && cd ../src/iri/
git clone https://github.com/albertpumarola/crawler_detector_gate.git
```
### Run Test
```
cd src/CrawlerDetector/
python detect_cam_no_ros.py
```
### Run ROS
Set path of weights in the launch file (<node>/src/CrawlerDetector/checkpoints)
```
roslaunch crawler_detector crawler_detector.launch 
```
### IMPORTANT
If the model was correclty loaded the following terminal msg will appear:
```
loaded net: ./checkpoints/pretrained_model/net_epoch_277_id_net.pth
```
else:
```
NOT!! loaded net: ./checkpoints/pretrained_model/net_epoch_0_id_net.pth
```
