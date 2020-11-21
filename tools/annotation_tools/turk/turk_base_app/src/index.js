/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import React from 'react';
import ReactDOM from 'react-dom';
import App from './App';
import * as serviceWorker from './serviceWorker';

// Code to correct the turk enviorment for react (don't remove)
let r = document.createElement('div')
r.id = "root"
document.getElementsByTagName('body')[0].prepend(r)

window.onload = ()=>{
  try {
    document.getElementById('submitButton').remove()
  } catch (error) {
    
  }
}

ReactDOM.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
  document.getElementById('root')
);

// If you want your app to work offline and load faster, you can change
// unregister() to register() below. Note this comes with some pitfalls.
// Learn more about service workers: https://bit.ly/CRA-PWA
serviceWorker.unregister();
