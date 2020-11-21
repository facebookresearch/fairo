"""
Copyright (c) Facebook, Inc. and its affiliates.

generate QR code image and start the webserver to display the image
usage: run python qr_web.py -info <info to put in the qr code image>
to start the webserver and generate the QR Code image.
add the flag -update (webserver is already running, just need to regenerate the QR Code image)
"""
import sys
from os import curdir, sep
import qrcode
from PIL import Image
from http.server import BaseHTTPRequestHandler, HTTPServer
import argparse
 
PORT = 8000

class HTTPHandler(BaseHTTPRequestHandler):
	def do_GET(self):
		"""respond to GET requests, either html or png request"""
		if (self.path.endswith(".png")):
			#handle a png image request
			self.send_response(200)
			self.send_header('Content-type','image/png')
			self.end_headers()
			with open(curdir + sep + self.path, 'rb') as file:
				self.wfile.write(file.read())
		else:
			#handle a html (any other) request with sending html back
			self.send_response(200)
			self.send_header('Content-type',"text/html")
			self.end_headers()
			content = '''
				<html><head><title>Minecraft QR Code Connect</title></head>
				<body>
					<h1>Minecraft Bot - QR Code Connect!</h1>
					<img src = "qrcode.png" alt="qr code">
				</body>
				</html>
				'''
			self.wfile.write(bytes(content,'utf8'))
		return

def makeQRCode(info):
	""" generate QR code that stores info """
	img = qrcode.make(info)
	img.save(curdir+sep+"qrcode.png","PNG")

def run():
	""" start the webserver at PORT on localhost """
	print("Connect using QR Code: http://localhost:8000")
	server_address = ('',PORT) 
	httpd = HTTPServer(server_address, HTTPHandler)
	#starts the webserver with the configs (specified port and HTTP handler)
	httpd.serve_forever()

if __name__ == "__main__":
	""" parse the arguments given- either need to start the webserver or just update the QR code """
	parser = argparse.ArgumentParser()
	parser.add_argument("-update",action="store_true")
	parser.add_argument("-info",action="store",default="")
	parser.add_argument("-port",type=int,action="store",default=8000)
	options = parser.parse_args()
	PORT = options.port
	makeQRCode(options.info)
	if not options.update:
		run()
	
