/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import './App.css';
import React from 'react';
import {SocketContext, socket} from './context/socket';
import Main from './component/main';
import { Routes, Route, BrowserRouter } from "react-router-dom";
import NavBar from './component/navbar';
import { SUBPATHS } from './constants/subpaths';

function App() {
  return (
    <SocketContext.Provider value={socket}>
      <BrowserRouter>
        <div className="App">
          <NavBar />
          {/* Routes for different pipeline */}
          <Routes>
            <Route path={SUBPATHS.HOME.key} element={<Main />} />
            <Route path={SUBPATHS.NLU.key} element={<div>NLU here!</div>} />
            <Route path={SUBPATHS.OTHER.key} element={<div>Something else here</div>} />
          </Routes>
        </div>
      </BrowserRouter>
    </SocketContext.Provider>
  
  );
}

export default App;
