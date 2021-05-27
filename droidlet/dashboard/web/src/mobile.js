import React from "react";
import ReactDOM from "react-dom";

import "bootstrap/dist/css/bootstrap.css";

import MobileMainPane from "./MobileMainPane.js";

let width = window.innerWidth;
let imageWidth = width / 2 - 25;

ReactDOM.render(<MobileMainPane />, document.getElementById("root"));
