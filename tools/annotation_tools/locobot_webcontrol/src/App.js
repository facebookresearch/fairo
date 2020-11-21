/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import React from 'react';
import LocobotControl from './LocobotControl'
import Video from './Video'
import './App.css'
import socketIOClient from "socket.io-client";

const ENDPOINT = "http://0.0.0.0:7559";

let socket = socketIOClient(ENDPOINT)

function App() {
  return (
    <div>
      <Video ip={ENDPOINT}></Video>
      <LocobotControl socket={socket}/>
    </div>
  );
}

export default App;
