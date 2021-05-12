/* 
Copyright (c) Facebook, Inc. and its affiliates.

Class for managing and flowing video masks 
*/

var jsfeat = window.jsfeat; // Library doesn't support node style imports, this pulls the library from a old style import in index.html

export class VideoMask {
  constructor(video, fps = 30) {
    this.video = video;
    this.fps = fps;

    this.canvas = new OffscreenCanvas(
      this.video.videoWidth,
      this.video.videoHeight
    );
    this.ctx = this.canvas.getContext("2d");

    this.groundTruthTimes = [];
    console.log(Math.ceil(video.duration * this.fps));
    this.points = new Array(Math.ceil(video.duration * fps));
    this.length = Math.ceil(video.duration * fps);

    this.getImageDataAsync = this.getImageDataAsync.bind(this);
    this.flow = this.flow.bind(this);
  }

  addGroundTruthPoints(pts, time) {
    let index = this.processTime(time);
    this.points[index] = pts.flatMap((p) => {
      return [
        Math.min(Math.max(p.x, 1), this.video.videoWidth),
        Math.min(Math.max(p.y, 1), this.video.videoHeight),
      ];
    });
    this.groundTruthTimes.push(time);
  }

  async flowUntilFailure(time, direction = 1) {
    let pts = this.getPoints(time);
    if (pts.length === 0) return;

    let index = this.processTime(time);
    let k = index;

    do {
      var d = await this.flow(k, direction);
      k += direction;
      console.log(d.status);
      this.points[k] = d.pts;
    } while (
      d.status.filter((i) => {
        return i !== 1;
      }).length === 0 &&
      k < this.length - 1 &&
      k > 0
    );

    this.video.currentTime = k / this.fps;
  }

  async flowUntilCount(time, frames, direction = 1) {
    let pts = this.getPoints(time);
    if (pts.length === 0) return;

    let index = this.processTime(time);
    let k = index;

    do {
      var d = await this.flow(k, direction);
      k += direction;
      console.log(d.status);
      this.points[k] = d.pts;
    } while (Math.abs(k - index) < frames && k < this.length - 1 && k > 0);

    this.video.currentTime = k / this.fps;
  }

  getPoints(time) {
    let index = this.processTime(time);
    let rawPoints = this.points[index];
    if (!rawPoints) {
      return [];
    }
    let mapped = [];
    for (let i = 0; i < rawPoints.length; i += 2) {
      mapped.push({
        x: rawPoints[i],
        y: rawPoints[i + 1],
      });
    }
    return mapped;
  }

  processTime(time) {
    return Math.floor(time * this.fps);
  }

  async flow(i, direction = 1) {
    let img_pyr1, img_pyr2, point_count, point_status, xy1, xy2;

    point_count = this.points[i].length / 2;
    point_status = new Uint8Array(point_count);
    xy1 = new Float32Array(this.points[i]);
    xy2 = new Float32Array(point_count * 2);

    img_pyr1 = new jsfeat.pyramid_t(3);
    img_pyr2 = new jsfeat.pyramid_t(3);
    img_pyr1.allocate(
      this.video.videoWidth,
      this.video.videoHeight,
      jsfeat.U8_t | jsfeat.C1_t
    );
    img_pyr2.allocate(
      this.video.videoWidth,
      this.video.videoHeight,
      jsfeat.U8_t | jsfeat.C1_t
    );

    let imageData1 = await this.getImageDataAsync(i);
    let imageData2 = await this.getImageDataAsync(i + direction);

    if (!imageData1 || !imageData2) {
      //Todo handle video not buffered in time
    }

    jsfeat.imgproc.grayscale(
      imageData1.data,
      this.video.videoWidth,
      this.video.videoHeight,
      img_pyr1.data[0]
    );
    jsfeat.imgproc.grayscale(
      imageData2.data,
      this.video.videoWidth,
      this.video.videoHeight,
      img_pyr2.data[0]
    );

    img_pyr1.build(img_pyr1.data[0], true);
    img_pyr2.build(img_pyr2.data[0], true);

    jsfeat.optical_flow_lk.track(
      img_pyr1,
      img_pyr2,
      xy1,
      xy2,
      point_count,
      20,
      30,
      point_status,
      0.01,
      0.001
    );

    return {
      pts: xy2,
      status: point_status,
    };
  }

  getImageData(i) {
    this.video.currentTime = i / this.fps;
    console.log(this.video.readyState);
    if (this.video.readyState >= this.video.HAVE_CURRENT_DATA) {
      this.ctx.drawImage(this.video, 0, 0);
      return this.ctx.getImageData(
        0,
        0,
        this.video.videoWidth,
        this.video.videoHeight
      );
    } else {
      console.log("NotReady in flow");
      return null;
    }
  }

  async getImageDataAsync(i) {
    return new Promise((resolve, reject) => {
      this.video.currentTime = i / this.fps;
      var interval = setInterval(() => {
        if (this.video.readyState >= this.video.HAVE_CURRENT_DATA) {
          this.ctx.drawImage(this.video, 0, 0);
          clearInterval(interval);
          resolve(
            this.ctx.getImageData(
              0,
              0,
              this.video.videoWidth,
              this.video.videoHeight
            )
          );
        }
      }, 50);
    });
  }
}
