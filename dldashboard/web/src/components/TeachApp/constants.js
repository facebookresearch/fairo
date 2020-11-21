/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
export let BACKEND_URL;

if (process.env.REACT_APP_API_URL) {
  BACKEND_URL = process.env.REACT_APP_API_URL;
} else {
  BACKEND_URL = "http://localhost:9000";
}
