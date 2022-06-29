/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import './App.css';
import 'antd/dist/antd.css';
import './index.css'
import React from 'react';
import {SocketContext, socket} from './context/socket';
import Main from './component/main';
import { Routes, Route, BrowserRouter } from "react-router-dom";
import NavBar from './component/navbar';
import { SUBPATHS } from './constants/subpaths';
import {Layout} from 'antd';
import PipelinePanel from './component/pipeline/panel';

const {Header, Footer} = Layout;

function App() {
  return (
    <SocketContext.Provider value={socket}>
      <BrowserRouter>
        <div className="App">
          <Layout>
            <Header className="header">
              <NavBar />
            </Header>
            {/* Routes for different pipeline */}
            <Layout>
              <Routes>
                <Route path={SUBPATHS.HOME.key} element={<Main />} />
                <Route path={SUBPATHS.NLU.key} element={<PipelinePanel pipelineType={SUBPATHS.NLU} />}>
                  <Route path=":batch_id" element={<PipelinePanel pipelineType={SUBPATHS.NLU.label}/>}/>
                </Route>
                <Route path={SUBPATHS.OTHER.key} element={<div>Something else here</div>} />
              </Routes>
            </Layout>
            <Footer
              style={{
                textAlign: 'center',
              }}
            >
              Droidlet HITL Dashboard ©2022
            </Footer>
          </Layout>
        </div>
      </BrowserRouter>
    </SocketContext.Provider>
  
  );
}

export default App;
