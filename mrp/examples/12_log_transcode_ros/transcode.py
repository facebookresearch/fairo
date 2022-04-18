#!/usr/bin/env python
import os
import a0
from PIL import Image
import fairomsg
import json
import argparse

sensor_msgs = fairomsg.get_msgs('sensor_msgs')
img_builder = sensor_msgs.Image()

def load_capnp_class(pkt):
        header = dict(pkt.headers)
        payload=pkt.payload

        #load capnp class
        if 'rostype' in header:
            rostype = header['rostype']
            msg, cls = rostype.split('/')
            clazz=getattr(fairomsg.get_msgs(msg), cls)

            #decode payload
            cls_inst = clazz.from_bytes(payload)
            cls_dict = cls_inst.to_dict()
        else:
            cls_dict = {'rawpkt': payload.decode('ascii')}

        return header, cls_dict

def save_image(rospkt, fname):
    px_size = rospkt['step']//rospkt['width']
    if px_size == 3:
        img = Image.frombytes('RGB', (rospkt['width'], rospkt['height']), rospkt['data'])
    else:
        img = Image.frombytes('L', (rospkt['width'], rospkt['height']), rospkt['data'])
    img.save(fname)

def save_raw(pkt, fname):
    with open(fname, 'w') as f:
        f.write(json.dumps(pkt, indent=4))

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('log_root_folder')
    parser.add_argument('export_root_folder')
    args = parser.parse_args()

    for folder, subfolders, files in os.walk(args.log_root_folder):
        for f in files:
            fpath = os.path.join(folder, f)
            topic = fpath.split('@')[0]

            export_folder = os.path.join(args.export_root_folder, topic)
            os.makedirs(export_folder, exist_ok=True)
            print("FILE :", fpath)
            reader = a0.ReaderSync(a0.File(os.path.abspath(fpath)), a0.INIT_OLDEST)
            while reader.can_read():
                pkt = reader.read()
                header, ros_pkt_dict = load_capnp_class(pkt)

                print(f'{fpath}: {header}')
                pkt_file_prefix = os.path.join(export_folder, header['a0_transport_seq'])
                if 'image' in topic:
                    save_image(ros_pkt_dict, pkt_file_prefix + ".jpg")
                    ros_pkt_dict_copy = ros_pkt_dict.copy()
                    ros_pkt_dict_copy['data']  = '<remove>'
                    save_raw({'rospkt': ros_pkt_dict_copy, 'a0_header': header}, pkt_file_prefix + ".txt")
                else:
                    save_raw({'rospkt': ros_pkt_dict, 'a0_header': header}, pkt_file_prefix + ".txt")



main()
