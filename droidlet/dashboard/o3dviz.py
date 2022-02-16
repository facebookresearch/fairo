import os
import time

os.environ["WEBRTC_IP"] = "0.0.0.0"
os.environ["WEBRTC_PORT"] = "8889"

import open3d as o3d
o3d.visualization.webrtc_server.enable_webrtc()

from open3d.visualization import O3DVisualizer, gui
import threading
import queue


class O3dViz(threading.Thread):
    def __init__(self, *args, **kwargs):
        self.q = queue.Queue()
        super().__init__(*args, **kwargs)

    def put(self, name, command, obj):
        # pass
        self.q.put([name, command, obj])

    def run(self):        
        app = gui.Application.instance

        app.initialize()
        w = O3DVisualizer("o3dviz", 1024, 768)
        w.set_background((0.0, 0.0, 0.0, 1.0), None)
         
        app.add_window(w)
        reset_camera = False

        while True:
            app.run_one_tick()
            time.sleep(0.001)

            try:
                name, command, geometry = self.q.get_nowait()

                try:
                    if command == 'remove':
                        w.remove_geometry(name)
                    elif command == 'replace':
                        w.remove_geometry(name)
                        w.add_geometry(name, geometry)
                    elif command == 'add':
                        w.add_geometry(name, geometry)

                except:
                    print("failed to add geometry to scene")
                if not reset_camera:
                    # Look at A from camera placed at B with Y axis
                    # pointing at C
                    # useful for pyrobot co-ordinates
                    w.scene.camera.look_at([1, 0, 0],
                                           [-5, 0, 1],
                                           [0, 0, 1])
                    # useful for initial camera co-ordinates
                    # w.scene.camera.look_at([0, 0, 1],
                    #                        [0, 0, -1],
                    #                        [0, -1, 0])
                    reset_camera = True
                w.post_redraw()
            except queue.Empty:
                pass


o3dviz = O3dViz()

def start():
    o3dviz.start()
