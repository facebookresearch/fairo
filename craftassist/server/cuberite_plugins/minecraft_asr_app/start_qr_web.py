"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import sys
from os import curdir, sep
import qrcode
from PIL import Image
from http.server import BaseHTTPRequestHandler, HTTPServer
 
PORT = 8000

class HTTPHandler(BaseHTTPRequestHandler):
	def do_GET(self):
		if (self.path.endswith(".png")):
			self.send_response(200)
			self.send_header('Content-type','image/png')
			self.end_headers()
			with open(curdir+sep+self.path, 'rb') as file:
				self.wfile.write(file.read())
		else:
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
	img = qrcode.make(info)
	img.save(curdir+sep+"qrcode.png","PNG")

def run():
	server_address = ('',8000) 
	httpd = HTTPServer(server_address, HTTPHandler)
	httpd.serve_forever()

if __name__ == "__main__":
	info = sys.argv[1]
	makeQRCode(info)
	run()
	
