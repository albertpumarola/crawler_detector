# Aeroarms Crawler Detector

### Install
```
git clone https://github.com/albertpumarola/crawler_detector.git
cd crawler_detector
pip install -r requirements.txt
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
