import csv
import threading
import picamera
import spidev, time
import Adafruit_DHT as dht # for dht11 sensor
import datetime as dt

def analog_read(channel):
	r = spi.xfer2([1,(0x08 + channel) << 4, 0])
	adc_out = (((r[1]&0x03) << 8) + r[2])
	return adc_out

def Convert_volt(data):
	volt = data * 5.0 / 1024
	return volt 

def Convert_LM35DZ(data, channel):
	temp = (data*100*100)/float(1023)
	temp = round(temp, channel)
	return temp

spi = spidev.SpiDev()
spi.open(0,0)
spi.max_speed_hz = 1000000

slice_time = 0.01
serial_num = 1

pir_channel = 0
light_channel = 1
temp_channel = 2

pir_data = list()
light_data = list()
temp_data = list()

with picamera.PiCamera() as camera:

	start = dt.datetime.now()

	camera.start_preview()
	camera.start_recording("Video_" + str(start) + ".h264")
	camera.annotate_background = picamera.Color('black')

	PIR_fn = "Data_PIR_" + str(start) + ".txt"
	Light_fn = "Data_Light_" + str(start) + ".txt"
	Temp_fn = "Data_Temp_" + str(start) + ".txt"

	f1 = open(PIR_fn,"w")
	f2 = open(Light_fn,"w")
	f3 = open(Temp_fn,"w")

	start_time = time.time()

	while(True):
		try:
			cur_time = time.time()

			if cur_time - start_time >= slice_time:

				start_time = cur_time
				pir = analog_read(pir_channel)
				pir_vol = Convert_volt(pir)

				light = analog_read(light_channel)
				light_vol = Convert_volt(light)

				tempeature = analog_read(temp_channel)
				tempeature_vol = Convert_volt(tempeature)

				print("PIR :: ADC = %s(%d) voltage = %fV" % (hex(pir), pir, pir_vol))
        		print("LIGHT :: ADC = %s(%d) voltage = %fV" % (hex(light), light, light_vol))
	        	print("TEMP :: ADC = %s(%d) voltage = %fV" % (hex(tempeature), tempeature, tempeature_vol))

				print("Serial Number :: %d" % (serial_num))
				print("")

				f1.write(pir + " ")
				f2.write(light + " ")
				f3.write(tempeature + " ")

				serial_num += 1

				pir_data.append(pir)
				light_data.append(light)
				temp_data.append(tempeature)

			if serial_num % 100 == 0:

				camera.annotate_text = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S\n')
				camera.annotate_text += ("PIR :: %d %d %d %d %d %d %d \n" % (pir_data[0],pir_data[1],pir_data[2],pir_data[3],pir_data[4],pir_data[5],pir_data[6])
										 + "LIGHT :: %d\n" % (light_data)
										 + "TEMP :: %d\n" % (tempeature)
										 + "Serial_num :: %d" % (serial_num))

			#	f1.write(str(pir_data))
			#	f2.write(str(light_data))
			#	f3.write(str(temp_data))

		except KeyboardInterrupt:
			break;

	f1.close()
	f2.close()
	f3.close()

	camera.stop_preview()
	camera.close()
