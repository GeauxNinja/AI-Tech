"""
Project:         Object Detection, using Pi5 and Arducam
Author:          Vic Gauthreaux / GeauxNinja
Date:            03/16/2025
Description:     This script captures video from the Raspberry Pi 5 camera, 
                 runs YOLOv8 object detection, displays CPU temperature, 
                 and streams to a web browser using Flask.
"""


from flask import Flask, Response
from picamera2 import Picamera2
import cv2
from ultralytics import YOLO
import os

""" commenting this out march 2025
# this grabs the temp of the cpu (and possibly more?)
def get_cpu_temperature():
#    temp = os.popen("vcgencmd measure_temp").readline()
#    return temp.replace("temp=", "").strip()
"""
# Function to get Raspberry Pi CPU temperature in Fahrenheit
def get_cpu_temperature():
    temp_celsius = float(os.popen("vcgencmd measure_temp").readline().replace("temp=", "").replace("'C", "").strip())
    temp_fahrenheit = (temp_celsius * 9/5) + 32  # Convert to Fahrenheit
    return f"{temp_fahrenheit:.2f}°F"
app = Flask(__name__)

# load YOLOv8 model
model = YOLO("yolov8n.pt")

# initialize camera
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (1280, 720)}))
picam2.start()

def generate_frames():
    while True:
        frame = picam2.capture_array()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform object detection
        results = model(frame_rgb)

        # Draw detections on the frame
        annotated_frame = results[0].plot()
        
        # Encode the frame as JPEG
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/temperature')
def temperature():
    return f"CPU Temperature: {get_cpu_temperature()}"
@app.route('/')
def index():
    return '''
<head>
    <title>AI-Pi5 Camera Stream</title>
    <style>
      body {
	padding: 20px;
        text-align: center;
        color:#f2b330;;
	background-color: #0c1c44;
      }
      img {
        border: 3px solid #f2b330;
        border-radius: 15px;
        display: block;
        margin: 0 auto;
      }
    </style>
  </head>
  <body>
    <h1>AI-Pi5 Cam Stream</h1>
    <p id="cpu-temp">Loading CPU Temperature...</p>
    <img src="/video_feed" width="80%">
    <script>
      function updateTemp() 
      {
        fetch('/temperature')
          .then(response => response.text())
          .then(data => document.getElementById("cpu-temp").innerText = data);
      }
        setInterval(updateTemp, 5000);  // Update temp every 5 seconds
        updateTemp();
    </script>
  </body>
</html>

'''

if __name__ == '__main__':
    app.run(host='10.1.1.44', port=5001, debug=False) # change port and addy to your situation/environment
