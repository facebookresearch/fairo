/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import io from "socket.io-client";
import Memory2D from "./components/Memory2D";
import MemoryList from "./components/MemoryList";
import LiveImage from "./components/LiveImage";
import Settings from "./components/Settings";
import LiveObjects from "./components/LiveObjects";
import LiveHumans from "./components/LiveHumans";
import History from "./components/History";
import InteractApp from "./components/Interact/InteractApp";
import VoxelWorld from "./components/VoxelWorld/VoxelWorld";
import Timeline from "./components/Timeline/Timeline";
import TimelineResults from "./components/Timeline/TimelineResults";
import TimelineDetails from "./components/Timeline/TimelineDetails";
import MobileMainPane from "./MobileMainPane";
import Retrainer from "./components/Retrainer";
import Navigator from "./components/Navigator";
import { isMobile } from "react-device-detect";
import MainPane from "./MainPane";
import AgentThinking from "./components/Interact/AgentThinking";
import Message from "./components/Interact/Message";
import TurkInfo from "./components/Turk/TurkInfo";

/**
 * The main state manager for the dashboard.
 * It connects to the backend via socket.io.
 * It drives all frontend components with new data from the backend.
 * It drives all backend commands from the frontend.
 * It persistently reconnects to the backend upon disconnection.
 *
 * The interface to the frontend is mainly via two methods:
 * 1. connect(c): a UI component can connect to the stateManager
 *                and the stateManager will subsequently handle state
 *                updates to this UI component
 * 2. restart(url): the stateManager can flush out and restart itself to
 *                  a different URL than default
 *
 * The interface to the backend is via socket.io events.
 * The main event of interests are:
 * "sensor_payload": this is a message that is received from the backend
 *                   with the latest sensor metadata. The stateManager
 *                   subsequently updates the frontend with this metadata.
 * "command": this is a raw command message that is sent to the backend
 *            (currently in Navigator.js, but should be
 *             refactored to be in this class).
 *
 * This class can be seen as a poor-man's Redux.js library, as Redux
 * would be overkill for this small project.
 */
class StateManager {
  refs = [];
  socket = null;
  default_url = "http://localhost:8000";
  connected = false;
  initialMemoryState = {
    objects: new Map(),
    humans: new Map(),
    lastChatActionDict: null,
    chats: [
      { msg: "", failed: false },
      { msg: "", failed: false },
      { msg: "", failed: false },
      { msg: "", failed: false },
      { msg: "", failed: false },
    ],
    timelineEvent: "",
    timelineEventHistory: [],
    timelineSearchResults: [],
    timelineDetails: [],
    timelineFilters: ["Perceive", "Dialogue", "Interpreter", "Memory"],
    timelineSearchPattern: "",
    agentType: "locobot",
    commandState: "idle",
    commandPollTime: 500,
    isTurk: false,
    agent_replies: [{}],
    last_reply: "",
  };
  session_id = null;

  constructor() {
    this.processMemoryState = this.processMemoryState.bind(this);
    this.setChatResponse = this.setChatResponse.bind(this);
    this.setLastChatActionDict = this.setLastChatActionDict.bind(this);
    this.setConnected = this.setConnected.bind(this);
    this.updateAgentType = this.updateAgentType.bind(this);
    this.forceErrorLabeling = this.forceErrorLabeling.bind(this);
    this.updateStateManagerMemory = this.updateStateManagerMemory.bind(this);
    this.keyHandler = this.keyHandler.bind(this);
    this.updateVoxelWorld = this.updateVoxelWorld.bind(this);
    this.setVoxelWorldInitialState = this.setVoxelWorldInitialState.bind(this);
    this.memory = JSON.parse(JSON.stringify(this.initialMemoryState)); // We want a clone
    this.processRGB = this.processRGB.bind(this);
    this.processDepth = this.processDepth.bind(this);
    this.processRGBDepth = this.processRGBDepth.bind(this);

    this.processObjects = this.processObjects.bind(this);
    this.showAssistantReply = this.showAssistantReply.bind(this);
    this.processHumans = this.processHumans.bind(this);

    this.processMap = this.processMap.bind(this);

    this.returnTimelineEvent = this.returnTimelineEvent.bind(this);

    this.onObjectAnnotationSave = this.onObjectAnnotationSave.bind(this);
    this.startLabelPropagation = this.startLabelPropagation.bind(this);
    this.labelPropagationReturn = this.labelPropagationReturn.bind(this);
    this.onSave = this.onSave.bind(this);
    this.saveAnnotations = this.saveAnnotations.bind(this);
    this.annotationRetrain = this.annotationRetrain.bind(this);
    this.goOffline = this.goOffline.bind(this);
    this.handleMaxFrames = this.handleMaxFrames.bind(this);

    // set turk related params
    const urlParams = new URLSearchParams(window.location.search);
    const turkExperimentId = urlParams.get("turk_experiment_id");
    const mephistoAgentId = urlParams.get("mephisto_agent_id");
    const turkWorkerId = urlParams.get("turk_worker_id");
    this.setTurkExperimentId(turkExperimentId);
    this.setMephistoAgentId(mephistoAgentId);
    this.setTurkWorkerId(turkWorkerId);

    // set default url to actual ip:port
    this.default_url = window.location.host;

    let url = localStorage.getItem("server_url");
    if (url === "undefined" || url === undefined || url === null) {
      url = this.default_url;
    }
    this.setUrl(url);

    this.fps_time = performance.now();

    // Assumes that all socket events for a frame are received before the next frame
    this.curFeedState = {
      rgbImg: null,
      depth: null,
      objects: [], // Can be changed by annotation tool
      origObjects: [], // Original objects sent from backend
      pose: null,
    };
    this.prevFeedState = {
      rgbImg: null,
      depth: null,
      objects: [],
      pose: null,
    };
    this.stateProcessed = {
      rgbImg: false,
      depth: false,
      objects: false,
      pose: false,
    };
    this.frameCount = 0; // For filenames when saving
    this.categories = new Set();
    this.properties = new Set();
    this.annotationsSaved = true;
    this.offline = false;
    this.frameId = 0; // Offline frame count
    this.offlineObjects = {}; // Maps frame ids to masks
    this.updateObjects = [false, false]; // Update objects on the frame after the rgb image changes
    this.useDesktopComponentOnMobile = true; // switch to use either desktop or mobile annotation on mobile device
    // TODO: Finish mobile annotation component (currently UI is finished, not linked up with backend yet)
  }

  setDefaultUrl() {
    localStorage.removeItem("server_url");
    this.setUrl(this.default_url);
  }

  setUrl(url) {
    this.url = url;
    localStorage.setItem("server_url", url);
    if (this.socket) {
      this.socket.removeAllListeners();
    }
    this.restart(this.url);
  }

  setTurkExperimentId(turk_experiment_id) {
    localStorage.setItem("turk_experiment_id", turk_experiment_id);
  }

  getTurkExperimentId() {
    return localStorage.getItem("turk_experiment_id");
  }

  setMephistoAgentId(mephisto_agent_id) {
    localStorage.setItem("mephisto_agent_id", mephisto_agent_id);
  }

  getMephistoAgentId() {
    return localStorage.getItem("mephisto_agent_id");
  }

  setTurkWorkerId(turk_worker_id) {
    localStorage.setItem("turk_worker_id", turk_worker_id);
  }

  getTurkWorkerId() {
    return localStorage.getItem("turk_worker_id");
  }

  restart(url) {
    this.socket = io.connect(url, {
      transports: ["polling", "websocket"],
    });
    const socket = this.socket;
    // on reconnection, reset the transports option, as the Websocket
    // connection may have failed (caused by proxy, firewall, browser, ...)
    socket.on("reconnect_attempt", () => {
      socket.io.opts.transports = ["polling", "websocket"];
    });

    socket.on("connect", (msg) => {
      let ipAddress = "";
      async function getIP() {
        const response = await fetch("https://api.ipify.org/?format=json");
        const data = await response.json();
        return data;
      }
      getIP().then((data) => {
        ipAddress = data["ip"];
        const dateString = (+new Date()).toString(36);
        this.session_id = ipAddress + ":" + dateString; // generate session id from ipAddress and date of opening webapp
        console.log("session id is");
        console.log(this.session_id);
        this.socket.emit("store session id", this.session_id);
      });
      console.log("connect event");
      this.setConnected(true);
      this.socket.emit("get_memory_objects");
      this.socket.emit("get_agent_type");
    });

    socket.on("reconnect", (msg) => {
      console.log("reconnect event");
      this.setConnected(true);
      this.socket.emit("get_memory_objects");
      this.socket.emit("get_agent_type");
    });

    socket.on("disconnect", (msg) => {
      console.log("disconnect event");
      this.setConnected(false);
      this.memory = this.initialMemoryState;
      // clear state of all components
      this.refs.forEach((ref) => {
        if (!(ref instanceof TimelineDetails)) {
          ref.setState(ref.initialState);
          ref.forceUpdate();
        }
      });
      console.log("disconnected");
    });

    socket.on("setChatResponse", this.setChatResponse);
    socket.on("setLastChatActionDict", this.setLastChatActionDict);
    socket.on("memoryState", this.processMemoryState);
    socket.on("updateState", this.updateStateManagerMemory);
    socket.on("updateAgentType", this.updateAgentType);

    socket.on("rgb", this.processRGB);
    socket.on("depth", this.processDepth);
    socket.on("image", this.processRGBDepth); // RGB + Depth
    socket.on("objects", this.processObjects);
    socket.on("updateVoxelWorldState", this.updateVoxelWorld);
    socket.on("setVoxelWorldInitialState", this.setVoxelWorldInitialState);
    socket.on("showAssistantReply", this.showAssistantReply);
    socket.on("humans", this.processHumans);
    socket.on("map", this.processMap);
    socket.on("newTimelineEvent", this.returnTimelineEvent);
    socket.on("labelPropagationReturn", this.labelPropagationReturn);
    socket.on("annotationRetrain", this.annotationRetrain);
    socket.on("saveRgbSegCallback", this.saveAnnotations);
    socket.on("handleMaxFrames", this.handleMaxFrames);
  }

  updateStateManagerMemory(data) {
    /**
     * This function sets the statemanager memory state
     * to be what's on the server and force re-renders
     * components.
     */
    this.memory = data.memory;
    this.refs.forEach((ref) => {
      ref.forceUpdate();
    });
  }

  updateAgentType(data) {
    // Sets stateManager agentType to match backend and passes to MainPane
    this.memory.agentType = data.agent_type;
    this.refs.forEach((ref) => {
      if (ref instanceof MainPane) {
        ref.setState({ agentType: this.memory.agentType });
      }
      if (ref instanceof InteractApp) {
        ref.setState({ agentType: this.memory.agentType });
      }
    });
  }

  forceErrorLabeling(status) {
    // If TurkInfo successfully mounts, this is a HIT
    // Forced error labeling and start button prompt should be on
    this.memory.isTurk = status;
    this.memory.agent_replies = [
      { msg: "Click the 'Start' button to begin!", timestamp: Date.now() },
    ];
    this.refs.forEach((ref) => {
      if (ref instanceof InteractApp) {
        ref.setState({
          isTurk: this.memory.isTurk,
          agent_replies: this.memory.agent_replies,
        });
      }
      if (ref instanceof Message) {
        ref.setState({
          agent_replies: this.memory.agent_replies,
        });
      }
    });
  }

  setConnected(status) {
    this.connected = status;
    this.refs.forEach((ref) => {
      // this has a potential race condition
      // (i.e. ref is not registered by the time socketio connects)
      // hence, in ref componentDidMount, we also
      // check set connected state
      if (ref instanceof Settings) {
        ref.setState({ connected: status });
      }
      if (ref instanceof Message) {
        ref.setState({ connected: status });
      }
    });
  }

  setChatResponse(res) {
    if (isMobile) {
      alert("Received text message: " + res.chat);
    }
    this.memory.chats = res.allChats;

    // Set the commandState to display 'received' for one poll cycle and then switch
    this.memory.commandState = "received";
    setTimeout(() => {
      if (this.memory.commandState === "received") {
        // May have moved on already, in which case leave it
        this.memory.commandState = "thinking";
      }
    }, this.memory.commandPollTime - 1); // avoid race condition

    // once confirm that this chat has been sent, clear last action dict
    this.memory.lastChatActionDict = null;

    this.refs.forEach((ref) => {
      if (ref instanceof InteractApp) {
        ref.setState({
          status: res.status,
        });
      }
      if (ref instanceof History) {
        ref.forceUpdate();
      }
    });
  }

  setLastChatActionDict(res) {
    this.memory.lastChatActionDict = res.action_dict;
  }

  updateVoxelWorld(res) {
    this.refs.forEach((ref) => {
      if (ref instanceof VoxelWorld) {
        ref.setState({
          world_state: res.world_state,
          status: res.status,
        });
      }
    });
  }

  setVoxelWorldInitialState(res) {
    this.refs.forEach((ref) => {
      if (ref instanceof VoxelWorld) {
        console.log("set Voxel World Initial state: " + res.world_state);
        ref.setState({
          world_state: res.world_state,
          status: res.status,
        });
      }
    });
  }

  showAssistantReply(res) {
    this.memory.agent_replies.push({
      msg: res.agent_reply,
      timestamp: Date.now(),
    });
    this.memory.last_reply = res.agent_reply;
    this.refs.forEach((ref) => {
      if (ref instanceof InteractApp) {
        ref.setState({
          agent_replies: this.memory.agent_replies,
        });
      }
      if (ref instanceof Message) {
        ref.setState({
          agent_replies: this.memory.agent_replies,
        });
      }
    });
  }

  sendCommandToTurkInfo(cmd) {
    this.refs.forEach((ref) => {
      if (ref instanceof TurkInfo) {
        ref.calcCreativity(cmd);
      }
    });
  }

  updateTimeline() {
    this.refs.forEach((ref) => {
      if (ref instanceof Timeline) {
        ref.forceUpdate();
      }
    });
  }

  returnTimelineEvent(res) {
    this.memory.timelineEventHistory.push(res);
    this.memory.timelineEvent = res;
    this.updateTimeline();

    // If the agent has finished processing the command
    // notify the user to look for an empty task stack
    if (JSON.parse(res).name === "perceive") {
      this.memory.commandState = "done_thinking";
      this.refs.forEach((ref) => {
        if (ref instanceof AgentThinking) {
          ref.sendTaskStackPoll(); // Do this once from here
        }
      });
    }
    // If there's an action to take in the world,
    // notify the user that it's executing
    if (JSON.parse(res).name === "interpreter") {
      this.memory.commandState = "executing";
    }
  }

  updateTimelineResults() {
    this.refs.forEach((ref) => {
      if (ref instanceof TimelineResults) {
        ref.forceUpdate();
      }
    });
  }

  keyHandler(key_codes) {
    let commands = [];
    let keys = [];
    for (var k in key_codes) {
      let val = key_codes[k];
      k = parseInt(k);
      keys.push(k);
      if (val === true) {
        if (k === 38) {
          // Up
          commands.push("MOVE_FORWARD");
        }
        if (k === 40) {
          // Down
          commands.push("MOVE_BACKWARD");
        }
        if (k === 37) {
          // Left
          commands.push("MOVE_LEFT");
        }
        if (k === 39) {
          // Right
          commands.push("MOVE_RIGHT");
        }
        if (k === 49) {
          // 1
          commands.push("PAN_LEFT");
        }
        if (k === 50) {
          // 2
          commands.push("PAN_RIGHT");
        }
        if (k === 51) {
          // 3
          commands.push("TILT_UP");
        }
        if (k === 52) {
          // 4
          commands.push("TILT_DOWN");
        }
      }
    }
    if (commands.length > 0) {
      let movementValues = {};
      this.refs.forEach((ref) => {
        if (ref instanceof Navigator) {
          movementValues = ref.state;
        }
      });

      this.socket.emit("movement command", commands, movementValues);

      // Reset keys to prevent duplicate commands
      for (let i in keys) {
        key_codes[keys[i]] = false;
      }
    }
  }

  /**
   * sends commands to backend for mobile button navigation
   * similar to keyHandler, but keyHandler is for web and arrow keys
   */
  buttonHandler(commands) {
    if (commands.length > 0) {
      this.socket.emit("movement command", commands);
    }
  }

  /**
   * key and value is the key value pair to be logged by flask
   * into interaction_loggings.json
   */
  logInteractiondata(key, value) {
    let interactionData = {};
    interactionData["session_id"] = this.session_id;
    interactionData["mephisto_agent_id"] = this.getMephistoAgentId();
    interactionData["turk_worker_id"] = this.getTurkWorkerId();
    interactionData[key] = value;
    this.socket.emit("interaction data", interactionData);
  }

  onObjectAnnotationSave(res) {
    // Process annotations
    let { nameMap, pointMap, propertyMap } = res;
    let newObjects = [];
    let scale = 500; // hardcoded from somewhere else
    for (let id in nameMap) {
      let oldObj = id < this.curFeedState.objects.length;
      let newId = oldObj ? this.curFeedState.objects[id].id : null;
      let newXyz = oldObj ? this.curFeedState.objects[id].xyz : null;
      // Get rid of masks with <3 points
      // We have this check because detector sometimes sends masks with <3 points to frontend
      let i = 0;
      while (i < pointMap[id].length) {
        if (!pointMap[id][i] || pointMap[id][i].length < 3) {
          pointMap[id].splice(i, 1);
          continue;
        }
        i++;
      }
      let newMask = pointMap[id].map((mask) =>
        mask.map((pt, i) => [pt.x * scale, pt.y * scale])
      );
      let newBbox = this.getNewBbox(newMask);

      newObjects.push({
        label: nameMap[id],
        mask: newMask,
        properties: propertyMap[id].join("\n "),
        type: "annotate", // either "annotate" or "detector"
        id: newId,
        bbox: newBbox,
        xyz: newXyz,
      });
    }
    this.curFeedState.objects = newObjects;
    this.annotationsSaved = false;

    this.refs.forEach((ref) => {
      if (ref instanceof LiveObjects) {
        ref.setState({
          objects: this.curFeedState.objects,
          updateFixup: true,
        });
      }
    });
    if (this.offline) {
      this.offlineObjects[this.frameId] = this.curFeedState.objects;
    }
  }

  getNewBbox(maskSet) {
    let xs = [],
      ys = [];
    for (let i = 0; i < maskSet.length; i++) {
      for (let j = 0; j < maskSet[i].length; j++) {
        xs.push(maskSet[i][j][0]);
        ys.push(maskSet[i][j][1]);
      }
    }
    let minX = Math.min.apply(null, xs),
      maxX = Math.max.apply(null, xs),
      minY = Math.min.apply(null, ys),
      maxY = Math.max.apply(null, ys);
    return [minX, minY, maxX, maxY];
  }

  startLabelPropagation() {
    // Update categories and properties
    let prevObjects = this.prevFeedState.objects.filter(
      (o) => o.type === "annotate"
    );
    for (let i in prevObjects) {
      this.categories.add(prevObjects[i].label);
      let prevProperties = prevObjects[i].properties.split("\n ");
      prevProperties.forEach((p) => this.properties.add(p));
    }

    // Label prop
    if (prevObjects.length > 0) {
      let labelProps = {
        prevRgbImg: this.prevFeedState.rgbImg,
        depth: this.curFeedState.depth,
        prevDepth: this.prevFeedState.depth,
        objects: prevObjects,
        basePose: this.curFeedState.pose,
        prevBasePose: this.prevFeedState.pose,
      };
      this.socket.emit("label_propagation", labelProps);
    }

    // Save rgb/seg if needed
    if (!this.annotationsSaved) {
      let saveProps = {
        rgb: this.prevFeedState.rgbImg,
        objects: prevObjects,
        frameCount: this.frameCount,
        categories: [null, ...this.categories], // Include null so category indices start at 1
      };
      this.socket.emit("save_rgb_seg", saveProps);
      this.annotationsSaved = true;
    }
    // Reset
    this.stateProcessed.rgbImg = true;
    this.stateProcessed.depth = true;
    this.stateProcessed.objects = true;
    this.stateProcessed.pose = true;
    this.frameCount++;
  }

  labelPropagationReturn(res) {
    this.refs.forEach((ref) => {
      if (ref instanceof LiveObjects) {
        for (let i in res) {
          // Get rid of masks with <3 points
          let j = 0;
          while (j < res[i].mask.length) {
            if (!res[i].mask[j] || res[i].mask[j].length < 3) {
              res[i].mask.splice(j, 1);
              continue;
            }
            j++;
          }
          res[i].bbox = this.getNewBbox(res[i].mask);
          ref.addObject(res[i]);
          this.curFeedState.objects.push(res[i]);
        }
      }
    });
    if (this.offline) {
      this.offlineObjects[this.frameId] = this.curFeedState.objects;
    }
    if (Object.keys(res).length > 0) {
      this.annotationsSaved = false;
    }
  }

  checkRunLabelProp() {
    return (
      this.curFeedState.rgbImg &&
      this.curFeedState.depth &&
      this.curFeedState.objects.length > 0 &&
      this.curFeedState.pose &&
      this.prevFeedState.rgbImg &&
      this.prevFeedState.depth &&
      this.prevFeedState.objects.length > 0 &&
      this.prevFeedState.pose &&
      !this.stateProcessed.rgbImg &&
      !this.stateProcessed.depth &&
      !this.stateProcessed.objects &&
      !this.stateProcessed.pose
    );
  }

  onSave() {
    console.log("saving annotations, categories, and properties");

    if (this.offline) {
      // Save categories and properties
      for (let key in this.offlineObjects) {
        let objects = this.offlineObjects[key];
        for (let i in objects) {
          let obj = objects[i];
          this.categories.add(obj.label);
          let properties = obj.properties.split("\n ");
          properties.forEach((p) => this.properties.add(p));
        }
      }
      let categories = [null, ...this.categories]; // Include null so category indices start at 1
      let properties = [...this.properties];
      this.socket.emit("save_categories_properties", categories, properties);

      // Save rgb/seg
      let outputId = 0;
      for (let key in this.offlineObjects) {
        let objects = this.offlineObjects[key];
        let finalFrame =
          outputId === Object.keys(this.offlineObjects).length - 1;
        let saveProps = {
          filepath: this.filepath,
          frameId: parseInt(key),
          outputId,
          objects,
          categories,
          finalFrame, // When true, backend will save all annotations to COCO format
        };
        this.socket.emit("offline_save_rgb_seg", saveProps);
        outputId++;
      }
    } else {
      // Save current rgb/seg if needed
      if (!this.annotationsSaved) {
        let curObjects = this.curFeedState.objects.filter(
          (o) => o.type === "annotate"
        );
        for (let i in curObjects) {
          this.categories.add(curObjects[i].label);
          let props = curObjects[i].properties.split("\n ");
          props.forEach((p) => this.properties.add(p));
        }
        let saveProps = {
          rgb: this.curFeedState.rgbImg,
          objects: curObjects,
          frameCount: this.frameCount,
          categories: [null, ...this.categories], // Include null so category indices start at 1
          callback: true, // Include boolean param to save annotations after -- ensures whatever the noun form of synchronous is
        };
        // This emit has a callback that calls saveAnnotations()
        this.socket.emit("save_rgb_seg", saveProps);
        this.annotationsSaved = true;
      } else {
        this.saveAnnotations();
      }
    }
  }

  saveAnnotations() {
    // Save annotations
    let categories = [null, ...this.categories]; // Include null so category indices start at 1
    let properties = [...this.properties];
    this.socket.emit("save_annotations", categories);
    this.socket.emit("save_categories_properties", categories, properties);
  }

  annotationRetrain(res) {
    console.log("retrained!");
    this.refs.forEach((ref) => {
      if (ref instanceof LiveObjects || ref instanceof Retrainer) {
        ref.setState({
          modelMetrics: res,
        });
      }
    });
  }

  goOffline(filepath) {
    console.log("Going offline with filepath", filepath);
    this.filepath = filepath;
    this.frameId = 0;
    this.offline = true;

    this.socket.emit("get_offline_frame", {
      filepath: this.filepath,
      frameId: this.frameId,
    });
    this.socket.emit("start_offline_dashboard", filepath);
    this.refs.forEach((ref) => {
      if (ref instanceof LiveObjects) {
        ref.setState({
          objects: [],
          modelMetrics: null,
          offline: true,
        });
      }
    });
  }

  handleMaxFrames(maxFrames) {
    this.maxOfflineFrames = maxFrames;
    console.log("max frames:", maxFrames);
  }

  offlineLabelProp(srcFrame, curFrame) {
    // Get src frame's objects
    let srcObjects = this.offlineObjects[srcFrame];
    let props = {
      filepath: this.filepath,
      srcFrame,
      curFrame,
      objects: srcObjects,
    };

    // Send objs and id to backend
    this.socket.emit("offline_label_propagation", props);
  }

  previousFrame() {
    if (this.frameId === 0) {
      console.log("no frames under 0");
      return;
    }
    this.frameId--;
    console.log("Prev frame", this.frameId);
    this.socket.emit("get_offline_frame", {
      filepath: this.filepath,
      frameId: this.frameId,
    });
    // Get objects
    this.curFeedState.objects = this.offlineObjects[this.frameId] || [];
    this.refs.forEach((ref) => {
      if (ref instanceof LiveObjects) {
        ref.setState({
          objects: this.curFeedState.objects,
        });
      }
    });

    // Run label prop
    let curFrameHasObjects =
      this.offlineObjects[this.frameId] &&
      this.offlineObjects[this.frameId].length > 0;
    if (this.offlineObjects[this.frameId + 1] && !curFrameHasObjects) {
      this.offlineLabelProp(this.frameId + 1, this.frameId);
    }
  }

  nextFrame() {
    if (this.frameId === this.maxOfflineFrames) {
      console.log("no frames over", this.maxOfflineFrames);
      return;
    }
    this.frameId++;
    console.log("Next frame", this.frameId);
    this.socket.emit("get_offline_frame", {
      filepath: this.filepath,
      frameId: this.frameId,
    });
    this.curFeedState.objects = this.offlineObjects[this.frameId] || [];
    this.refs.forEach((ref) => {
      if (ref instanceof LiveObjects) {
        ref.setState({
          objects: this.curFeedState.objects,
        });
      }
    });

    // Run label prop
    let curFrameHasObjects =
      this.offlineObjects[this.frameId] &&
      this.offlineObjects[this.frameId].length > 0;
    if (this.offlineObjects[this.frameId - 1] && !curFrameHasObjects) {
      this.offlineLabelProp(this.frameId - 1, this.frameId);
    }
  }

  processMemoryState(msg) {
    this.refs.forEach((ref) => {
      if (ref instanceof MemoryList) {
        ref.setState({ isLoaded: true, memory: msg });
      }
    });
  }

  processRGB(res) {
    if (this.curFeedState.rgbImg !== res) {
      // Update feed
      let rgb = new Image();
      rgb.src = "data:image/webp;base64," + res;
      this.refs.forEach((ref) => {
        if (ref instanceof LiveImage) {
          if (ref.props.type === "rgb") {
            ref.setState({
              isLoaded: true,
              rgb: rgb,
            });
          }
        }
        if (this.offline && ref instanceof LiveObjects) {
          ref.setState({ rgb });
        }
      });
      // Update state
      this.prevFeedState.rgbImg = this.curFeedState.rgbImg;
      this.curFeedState.rgbImg = res;
      this.stateProcessed.rgbImg = false;
      this.updateObjects = [true, true]; // Change objects on frame after this one
    }
    if (this.checkRunLabelProp()) {
      this.startLabelPropagation();
    }
  }

  processDepth(res) {
    if (this.curFeedState.depth !== res) {
      // Update feed
      let depth = new Image();
      depth.src = "data:image/webp;base64," + res.depthImg;
      this.refs.forEach((ref) => {
        if (ref instanceof LiveImage) {
          if (ref.props.type === "depth") {
            ref.setState({
              isLoaded: true,
              depth: depth,
            });
          }
        }
      });
      // Update state
      this.prevFeedState.depth = this.curFeedState.depth;
      this.curFeedState.depth = {
        depthImg: res.depthImg,
        depthMax: res.depthMax,
        depthMin: res.depthMin,
      };
      this.stateProcessed.depth = false;
    }
    if (this.checkRunLabelProp()) {
      this.startLabelPropagation();
    }
  }

  processRGBDepth(res) {
    this.processRGB(res.rgb);
    this.processDepth(res.depth);
  }

  processObjects(res) {
    if (res.image === -1 || res.image === undefined) {
      return;
    }
    let rgb = new Image();
    rgb.src = "data:image/webp;base64," + res.image.rgb;

    // Get rid of empty masks
    let i = 0;
    while (i < res.objects.length) {
      let j = 0;
      while (j < res.objects[i].mask.length) {
        if (!res.objects[i].mask[j] || res.objects[i].mask[j].length < 3) {
          res.objects[i].mask.splice(j, 1);
          continue;
        }
        j++;
      }
      if (res.objects[i].mask.length === 0) {
        res.objects.splice(i, 1);
        continue;
      }
      i++;
    }
    res.objects.forEach((o) => {
      o["type"] = "detector";
    });

    // If new objects, update state and feed
    if (
      this.updateObjects[1] // Frame after rgb changes
    ) {
      this.prevFeedState.objects = this.curFeedState.objects;
      this.curFeedState.objects = JSON.parse(JSON.stringify(res.objects)); // deep clone
      this.curFeedState.origObjects = JSON.parse(JSON.stringify(res.objects)); // deep clone
      this.stateProcessed.objects = false;
      this.updateObjects = [false, false];

      this.refs.forEach((ref) => {
        if (ref instanceof LiveObjects) {
          ref.setState({
            objects: res.objects,
            rgb: rgb,
            height: res.height,
            width: res.width,
            scale: res.scale,
          });
        } else if (ref instanceof MobileMainPane) {
          // mobile main pane needs to know object_rgb so it can be passed into annotation image when pane switches to annotation
          ref.setState({
            objectRGB: rgb,
          });
        }
      });
    }
    if (this.updateObjects[0]) {
      // Current frame is when rgb changes. This is needed to ensure correctness
      this.updateObjects[1] = true;
    }
    if (this.checkRunLabelProp()) {
      this.startLabelPropagation();
    }
  }

  processHumans(res) {
    if (res.image === -1 || res.image === undefined) {
      return;
    }
    let rgb = new Image();
    rgb.src = "data:image/webp;base64," + res.image.rgb;

    this.refs.forEach((ref) => {
      if (ref instanceof LiveHumans) {
        ref.setState({
          isLoaded: true,
          humans: res.humans,
          rgb: rgb,
          height: res.height,
          width: res.width,
          scale: res.scale,
        });
      }
    });
  }

  processMap(res) {
    this.refs.forEach((ref) => {
      if (ref instanceof Memory2D) {
        ref.setState({
          isLoaded: true,
          memory: this.memory,
          bot_xyz: [res.x, res.y, res.yaw],
          obstacle_map: res.map,
        });
      }
    });

    if (
      !this.curFeedState.pose ||
      (res &&
        (res.x !== this.curFeedState.pose.x ||
          res.y !== this.curFeedState.pose.y ||
          res.yaw !== this.curFeedState.pose.yaw))
    ) {
      this.prevFeedState.pose = this.curFeedState.pose;
      this.curFeedState.pose = {
        x: res.x,
        y: res.y,
        yaw: res.yaw,
      };
      this.stateProcessed.pose = false;
    }
    if (this.checkRunLabelProp()) {
      this.startLabelPropagation();
    }
  }

  connect(o) {
    this.refs.push(o);
  }

  disconnect(o) {
    // remove the passed in parameter from the list of refs to prevent memory leaks
    this.refs = this.refs.filter(function (item) {
      return item !== o;
    });
  }
}
var stateManager = new StateManager();

// export a single reused stateManager object,
// rather than the class, so that it is reused across tests in the same lifetime
export default stateManager;
