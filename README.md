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
