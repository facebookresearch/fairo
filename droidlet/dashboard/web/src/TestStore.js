/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import Memory2D from "./components/Memory2D";
import LiveImage from "./components/LiveImage";
import MainPane from "./MainPane";

class TestStore {
  base_url = "https://locobot-bucket.s3-us-west-2.amazonaws.com/";
  json_url = "test_assets/json_memory_list.json";
  refs = [];
  memory = {
    objects: new Map(),
    humans: new Map(),
  };
  constructor() {
    this.updateMemory = this.updateMemory.bind(this);
    this.loopState = this.loopState.bind(this);

    fetch(this.base_url + this.json_url)
      .then((res) => res.json())
      .then((urls) => {
        this.urls = urls;
        this.cur_url = 0;
        this.looped_once = false;
        this.loopState();
      });
  }

  updateMemory(res) {
    res.objects.forEach((obj) => {
      let key = JSON.stringify(obj); // I'm horrible person for doing this!!!
      obj.xyz = [obj.xyz[0] + res.x, obj.xyz[1] + res.y, obj.xyz[2]];
      this.memory.objects.set(key, obj);
    });
  }

  loopState() {
    fetch(this.base_url + this.urls[this.cur_url])
      .then((res) => res.json())
      .then((res) => {
        let rgb = new Image();
        rgb.src = "data:image/png;base64," + res.image.rgb;
        let depth = new Image();
        depth.src = "data:image/png;base64," + res.image.depth;

        this.updateMemory(res);

        this.refs.forEach((ref) => {
          if (ref instanceof Memory2D) {
            ref.setState({
              isLoaded: true,
              memory: this.memory,
              bot_xyz: [res.x, res.y, res.yaw],
            });
          } else {
            ref.setState({
              isLoaded: true,
              rgb: rgb,
              depth: depth,
              objects: res.objects,
              humans: res.humans,
            });
          }
        });
        this.cur_url++;
        if (this.cur_url === this.urls.length) {
          this.cur_url = 0;
          this.looped_once = true;
        }
        setTimeout(this.loopState, 0);
      });
  }

  connect(o) {
    this.refs.push(o);
  }
}
var testStore = new TestStore();

// export a single reused testStore object,
// rather than the class, so that it is reused across tests in the same lifetime
export default testStore;
